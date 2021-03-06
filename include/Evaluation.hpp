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


#ifndef INCLUDE_EVALUATION_HPP_
#define INCLUDE_EVALUATION_HPP_

#include <vector>
#include <memory>
#include <thread>
#include <fstream>

#include <opencv2/opencv.hpp>

class EvaluationThread{
public:
	EvaluationThread(cv::Mat itrainEnvDescs,
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
				cv::Mat iclTree);

	~EvaluationThread();

	bool hasEnded();

	float getPzGe();

	float getPzGne();
private:
	cv::Mat trainEnvDescs;
	cv::Mat trainDescs;
	std::vector<int> trainBuildings;
	std::vector<int> trainFloors;
	std::vector<cv::Point2f> trainPoses;
	cv::Mat testDescs;
	std::vector<int> testBuildings;
	std::vector<int> testFloors;
	std::vector<cv::Point2f> testPoses;
	std::vector<int>& matchedTestIdxs;
	int& correct;
	int& incorrect;
	int& unrec;
	float PzGe;
	float PzGne;
	cv::Mat clTree;

	std::thread runThread;
	bool hasEndedFlag;

	static constexpr float matchProbThresh = 0.0;

	void evaluate();
};

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
					std::ofstream& resultsFile);

#endif /* INCLUDE_EVALUATION_HPP_ */
