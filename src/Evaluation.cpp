/*
	Copyright (c) 2016,	Mobile Robots Lab Team:
	-Michal Nowicki (michal.nowicki [at] put.poznan.pl),
	-Jan Wietrzykowski (jan.wietrzykowski [at] put.poznan.pl).
	Poznan University of Technology
	All rights reserved.

	This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <iostream>
#include <queue>

#include "Evaluation.hpp"
#include "openfabmap.hpp"

using namespace std;
using namespace cv;

EvaluationThread::EvaluationThread(cv::Mat itrainEnvDescs,
				cv::Mat itrainDescs,
				const std::vector<int>& itrainBuildings,
				const std::vector<int>& itrainFloors,
				const std::vector<cv::Point2f>& itrainPoses,
				cv::Mat itestDescs,
				const std::vector<int>& itestBuildings,
				const std::vector<int>& itestFloors,
				const std::vector<cv::Point2f>& itestPoses,
				std::vector<int>& imatchedTestIdxs,
				int& icorrect,
				int& iincorrect,
				int& iunrec,
				float iPzGe,
				float iPzGne,
				cv::Mat iclTree)
	: trainEnvDescs(itrainEnvDescs),
	  trainDescs(itrainDescs),
	  trainBuildings(itrainBuildings),
	  trainFloors(itrainFloors),
	  trainPoses(itrainPoses),
	  testDescs(itestDescs),
	  testBuildings(itestBuildings),
	  testFloors(itestFloors),
	  testPoses(itestPoses),
	  matchedTestIdxs(imatchedTestIdxs),
	  correct(icorrect),
	  incorrect(iincorrect),
	  unrec(iunrec),
	  PzGe(iPzGe),
	  PzGne(iPzGne),
	  clTree(iclTree.clone()),
	  hasEndedFlag(false)
{
	runThread = thread(&EvaluationThread::evaluate, this);
}

EvaluationThread::~EvaluationThread(){
	if(runThread.joinable()){
		runThread.join();
	}
}

bool EvaluationThread::hasEnded(){
	return hasEndedFlag;
}

float EvaluationThread::getPzGe(){
	return PzGe;
}

float EvaluationThread::getPzGne(){
	return PzGne;
}

void EvaluationThread::evaluate()
{
	shared_ptr<of2::FabMap> fabmap(new of2::FabMap2(clTree,
													PzGe,
													PzGne,
													of2::FabMap::SAMPLED | of2::FabMap::CHOW_LIU));

//			cout << "Adding train data" << endl;
	fabmap->addTraining(trainEnvDescs);

	cout << "Adding places database" << endl;
	fabmap->add(trainDescs);

	cout << "Matching" << endl;
	matchedTestIdxs.clear();
	correct = 0;
	incorrect = 0;
	unrec = 0;
	int all = 0;
	float distErr = 0.0;
	for(int d = 0; d < testDescs.rows; ++d){
		std::vector<of2::IMatch> matches;
		fabmap->compare(testDescs.row(d), matches, false);
		static constexpr int numBestMatches = 5;
		double bestMatchProb = 0.0;
		int bestMatchIdx = -1;
//		std::cout << "matches.size() = " << matches.size() << std::endl;
//		 lowest probability on top
//		priority_queue<pair<double, int>> matchesHeap;
		for(std::vector<of2::IMatch>::iterator it = matches.begin(); it != matches.end(); ++it){
//			cout << "it->match = " << it->match << endl;
			if(it->match > matchProbThresh){
				if(bestMatchProb < it->match){
					bestMatchProb = it->match;
					bestMatchIdx = it->imgIdx;
				}
			}
//			if(matchesHeap.size() < numBestMatches){
//				matchesHeap.push(make_pair(-it->match, it->imgIdx));
//			}
//			else{
//				if(-matchesHeap.top().first < it->match){
//					matchesHeap.pop();
//					matchesHeap.push(make_pair(-it->match, it->imgIdx));
//				}
//			}
		}


		matchedTestIdxs.push_back(bestMatchIdx);
//		std::cout << "Matched with " << bestMatchIdx << ", prob = " << bestMatchProb << std::endl;
		if(bestMatchIdx >= 0){
//						cout << "trainClusters[bestMatchIdx] = " << trainClusters[bestMatchIdx] << endl;
//						cout << "testClusters[d] = " << testClusters[d] << endl;
			if(trainBuildings[bestMatchIdx] == testBuildings[d] &&
				trainFloors[bestMatchIdx] == testFloors[d])
			{
//			if(trainClusters[bestMatchIdx] == testClusters[d])
//			{
				++correct;

				distErr += norm(trainPoses[bestMatchIdx] - testPoses[d]);
			}
			else{
				++incorrect;
			}
		}
		else{
			++unrec;
		}
	}
	cout << "PzGe = " << PzGe  << endl;
	cout << "PzGne = " << PzGne << endl;
	cout << "correct = " << correct << endl;
	cout << "incorrect = " << incorrect << endl;
	cout << "unrec = " << unrec << endl;
	cout << "success rate = " << (double)correct / (correct + incorrect + unrec) << endl;
	cout << "mean distError = " << distErr/correct << endl;

	hasEndedFlag = true;
}

void endThreadIfAny(std::vector<std::vector<std::shared_ptr<EvaluationThread>>>& threads,
					std::vector<std::vector<std::vector<int>>>& matchedTestIdxs,
					std::vector<std::vector<int>>& correct,
					std::vector<std::vector<int>>& incorrect,
					std::vector<std::vector<int>>& unrec,
					std::vector<std::vector<float>>& results,
					int& threadCnt,
					float& bestScore,
					float& bestPzge,
					float& bestPzgne,
					std::ofstream& resultsFile)
{
	for(int i = 0; i < threads.size(); ++i){
		for(int j = 0; j < threads[i].size(); ++j){
			if(threads[i][j]){
				if(threads[i][j]->hasEnded()){
					results[i][j] = (float)correct[i][j]/(correct[i][j] + incorrect[i][j] + unrec[i][j]);
					resultsFile << threads[i][j]->getPzGe() << " " << threads[i][j]->getPzGne() << " " << results[i][j] << endl;
					if(results[i][j] > bestScore){
						bestScore = results[i][j];
						bestPzge = threads[i][j]->getPzGe();
						bestPzgne = threads[i][j]->getPzGne();
					}
					matchedTestIdxs[i][j].clear();

					threads[i][j].reset();
					--threadCnt;
				}
			}
		}
	}
}
