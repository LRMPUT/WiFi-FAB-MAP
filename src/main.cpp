
#include <list>
#include <map>
#include <fstream>
#include <chrono>
#include <string>
#include <memory>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "openfabmap.hpp"
#include "Evaluation.hpp"

using namespace std;
using namespace cv;

static constexpr int descSize = 520;
static constexpr float eps = 1e-6;

template<class T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec){
	out << "[";
	for(int v = 0; v < (int)vec.size(); ++v){
		out << vec[v];
		if(v < vec.size() - 1){
			out << ", ";
		}
	}
	out << "]";

	return out;
}

int help(void)
{
	std::cout << "Usage: wififabmap [-s settingsfile]" << std::endl;
	return 0;
}


int loadDataFile(std::string filePath,
				cv::Mat& descs,
				std::vector<cv::Point2f>& poses,
				std::vector<int>& floors,
				std::vector<int>& buildings)
{
	ifstream dataFile(filePath);
	if(!dataFile.good()){
		cout << "Error - could not open file: " << filePath << endl;
	}
	string tmp;
	// discard first line with headers
	getline(dataFile, tmp);
	while(!dataFile.eof() && !dataFile.fail()){
//		cout << "descs.rows = " << descs.rows << endl;

		Mat curDesc(1, descSize, CV_32FC1, Scalar(0));
		for(int d = 0; d < descSize && !dataFile.fail(); ++d){
			getline(dataFile, tmp, ',');
			if(!dataFile.fail()){
				curDesc.at<float>(d) = stoi(tmp);
			}
		}
		Point2f curPose;
		if(!dataFile.fail()){
			getline(dataFile, tmp, ',');
//			cout << "tmp = " << tmp << endl;
			curPose.x = stof(tmp);
			getline(dataFile, tmp, ',');
//			cout << "tmp = " << tmp << endl;
			curPose.y = stof(tmp);
		}
		int curFloor;
		if(!dataFile.fail()){
			getline(dataFile, tmp, ',');
//			cout << "tmp = " << tmp << endl;
			curFloor = stoi(tmp);
		}
		int curBuilding;
		if(!dataFile.fail()){
			getline(dataFile, tmp, ',');
//			cout << "tmp = " << tmp << endl;
			curBuilding = stoi(tmp);
		}
		if(!dataFile.fail()){
			// discard rest of line
			getline(dataFile, tmp);
		}
		
		if(!dataFile.fail()){
			descs.push_back(curDesc);
			poses.push_back(curPose);
			floors.push_back(curFloor);
			buildings.push_back(curBuilding);
		}
	}
	
	return descs.rows;
}

int loadDataFileClust(std::string filePath,
				cv::Mat& descs,
				std::vector<cv::Point2f>& poses,
				std::vector<int>& floors,
				std::vector<int>& buildings,
				std::vector<int>& clusters)
{
	ifstream dataFile(filePath);
	if(!dataFile.good()){
		cout << "Error - could not open file: " << filePath << endl;
	}
	string tmp;
	// discard first line with headers
	getline(dataFile, tmp);
	while(!dataFile.eof() && !dataFile.fail()){
//		cout << "descs.rows = " << descs.rows << endl;

		Mat curDesc(1, descSize, CV_32FC1, Scalar(0));
		for(int d = 0; d < descSize && !dataFile.fail(); ++d){
			getline(dataFile, tmp, ',');
			if(!dataFile.fail()){
				curDesc.at<float>(d) = stoi(tmp);
			}
		}
		Point2f curPose;
		if(!dataFile.fail()){
			getline(dataFile, tmp, ',');
//			cout << "tmp = " << tmp << endl;
			curPose.x = stof(tmp);
			getline(dataFile, tmp, ',');
//			cout << "tmp = " << tmp << endl;
			curPose.y = stof(tmp);
		}
		int curFloor;
		if(!dataFile.fail()){
			getline(dataFile, tmp, ',');
//			cout << "tmp = " << tmp << endl;
			curFloor = stoi(tmp);
		}
		int curBuilding;
		if(!dataFile.fail()){
			getline(dataFile, tmp, ',');
//			cout << "tmp = " << tmp << endl;
			curBuilding = stoi(tmp);
		}
		if(!dataFile.fail()){
			// discard space id
			getline(dataFile, tmp, ',');
		}
		if(!dataFile.fail()){
			// discard relative position
			getline(dataFile, tmp, ',');
		}
		if(!dataFile.fail()){
			// discard user id
			getline(dataFile, tmp, ',');
		}
		if(!dataFile.fail()){
			// discard phone id
			getline(dataFile, tmp, ',');
		}
		if(!dataFile.fail()){
			// discard timestamp
			getline(dataFile, tmp, ',');
		}
		int curCluster;
		if(!dataFile.fail()){
			// cluster id
			getline(dataFile, tmp);
			curCluster = stoi(tmp);
		}

		if(!dataFile.fail()){
			descs.push_back(curDesc);
			poses.push_back(curPose);
			floors.push_back(curFloor);
			buildings.push_back(curBuilding);
			clusters.push_back(curCluster);
		}
	}

	return descs.rows;
}

void preprocessDescs(cv::Mat& descs){
	static constexpr int minVal = -110;
	static constexpr int maxVal = 0;
	static constexpr int stepVal = 5;
	static constexpr int notFoundVal = 100;

	descs -= Scalar(minVal);
//	Mat notFoundMask = (descs == notFoundVal - minVal);
//	descs.setTo(Scalar(0), notFoundMask);
	int binsPerDesc = ((maxVal - minVal)/stepVal);
	Mat extDescs(descs.rows, binsPerDesc * descs.cols, CV_32FC1, Scalar(0));
	for(int d = 0; d < descs.rows; ++d){
		for(int v = 0; v < descs.cols; ++v){
//			cout << "v = " << v << endl;
			if(fabs(descs.at<float>(d, v) - (notFoundVal - minVal)) > eps){
				float val = descs.at<float>(d, v);
//				std::default_random_engine gen;
//				std::uniform_int_distribution<int> dist(-30, 30);
//				val += dist(gen);
//				val = max(val, 0.0f);
//				val = min(val, (float)(maxVal - minVal));
				int bin = val/stepVal;
				for(int b = 0; b <= bin; ++b){
					extDescs.at<float>(d, v * binsPerDesc + b) = 1.0;
				}
//				cout << "val = " << val << endl;
//				cout << "index = " << v * binsPerDesc + bin << endl;
			}
		}
	}
	descs = extDescs;
}

void convertToMat(const std::vector<cv::Point2f>& src,
					const std::vector<int>& srcLevel,
					cv::Mat& dst)
{
	static constexpr float levelSep = 100;
	dst = Mat(0, 3, CV_32FC1, Scalar(0));
	for(int i = 0; i < src.size(); ++i){
		dst.push_back(Mat(Matx13f(src[i].x, src[i].y, srcLevel[i]*levelSep)));
	}
}

void assignClusters(const std::vector<cv::Point2f>& srcPoses,
					const std::vector<int>& srcFloors,
					const std::vector<int>& srcClust,
					const std::vector<cv::Point2f>& dstPoses,
					const std::vector<int>& dstFloors,
					std::vector<int>& dstClust,
					float distEps)
{
	Mat srcPosesMat;
	Mat dstPosesMat;
	cout << "Converting to MatVector" << endl;
	convertToMat(srcPoses, srcFloors, srcPosesMat);
	convertToMat(dstPoses, dstFloors, dstPosesMat);
//	cout << "srcPosesMat = " << srcPosesMat << endl;
//	cout << "dstPosesMat = " << dstPosesMat << endl;
	std::vector<DMatch> matches;
	BFMatcher matcher;
	cout << "Matching" << endl;
	matcher.match(dstPosesMat, srcPosesMat, matches);
	dstClust = vector<int>(dstPoses.size(), -1);
	cout << "Analyzing matches" << endl;
	for(int i = 0; i < matches.size(); ++i){
//		cout << "matches[" << i << "].distance = " << matches[i].distance << endl;
		if(matches[i].distance <= distEps){
			if(matches[i].trainIdx >= 0 && matches[i].trainIdx < srcClust.size()){
				dstClust[i] = srcClust[matches[i].trainIdx];
			}
		}
	}
}



int main(int argc, char * argv[]){

	string settfilename;
	if (argc == 1) {
		//assume settings in working directory
		settfilename = "settings.yml";
	}
	else if (argc == 3) {
		if(std::string(argv[1]) != "-s") {
			//incorrect option
			return help();
		} else {
			//settings provided as argument
			settfilename = std::string(argv[2]);
		}
	}
	else {
		//incorrect arguments
		return help();
	}
	cv::FileStorage fsSettings;
	fsSettings.open(settfilename, cv::FileStorage::READ);
	if (!fsSettings.isOpened()) {
		std::cerr << "Could not open settings file: " << settfilename << std::endl;
		return -1;
	}

	cv::Mat trainDescs;
	std::vector<cv::Point2f> trainPoses;
	std::vector<int> trainFloors;
	std::vector<int> trainBuildings;
	std::vector<int> trainClusters;
	
	loadDataFileClust("../res/trainingDataClust.csv",
						trainDescs,
						trainPoses,
						trainFloors,
						trainBuildings,
						trainClusters);

//	for(int d = 0; d < min(10, trainDescs.rows); ++d){
//		cout << trainDescs.row(d) << endl;
//	}

	preprocessDescs(trainDescs);

//	for(int d = 0; d < min(10, trainDescs.rows); ++d){
//		cout << trainDescs.row(d) << endl;
//	}

	cv::Mat trainEnvDescs;
	std::vector<cv::Point2f> trainEnvPoses;
	std::vector<int> trainEnvFloors;
	std::vector<int> trainEnvBuildings;
	std::vector<int> trainEnvClusters;

	loadDataFileClust("../res/trainingEnvDataClust.csv",
						trainEnvDescs,
						trainEnvPoses,
						trainEnvFloors,
						trainEnvBuildings,
						trainEnvClusters);

	preprocessDescs(trainEnvDescs);

	cv::Mat clTree;
	if((int)fsSettings["ChowLiuOptions"]["Train"] > 0){
		cout << "Learning Chow-Liu Tree" << endl;

		of2::ChowLiuTree tree;
		tree.add(trainEnvDescs);
		clTree = tree.make((double)fsSettings["ChowLiuOptions"]["LowerInfoBound"]);

		//save the resulting tree
		std::cout <<"Saving Chow-Liu Tree" << std::endl;
		FileStorage fsChowLiu(fsSettings["FilePaths"]["ChowLiuTree"], cv::FileStorage::WRITE);
		fsChowLiu << "ChowLiuTree" << clTree;
		fsChowLiu.release();
	}
	else{
		//load a chow-liu tree
		std::cout << "Loading Chow-Liu Tree" << std::endl;
		FileStorage fsChowLiu(fsSettings["FilePaths"]["ChowLiuTree"], cv::FileStorage::READ);
		fsChowLiu["ChowLiuTree"] >> clTree;
		if (clTree.empty()) {
			std::cout << (string)fsSettings["FilePaths"]["ChowLiuTree"] << ": Chow-Liu tree not found" << std::endl;
		}
		fsChowLiu.release();
	}

	static constexpr bool tuneParams = true;
	if(tuneParams){
		// random permutation for splitting training and validation set
		std::default_random_engine gen;
		std::vector<int> perm;
		for(int i = 0; i < trainDescs.rows; ++i){
			perm.push_back(i);
		}
		for(int i = 1; i < perm.size(); ++i){
			std::uniform_int_distribution<int> dist(0, i);
			int swapIdx = dist(gen);
			swap(perm[i], perm[swapIdx]);
		}
//		cout << perm << endl;

		static constexpr float valPart = 0.33;
		int trainStart = perm.size() * valPart;

		cv::Mat testSelDescs(0, trainDescs.cols, CV_32FC1);
		std::vector<int> testSelBuildings;
		std::vector<int> testSelFloors;
		std::vector<cv::Point2f> testSelPoses;
		for(int i = 0; i < trainStart; ++i){
			testSelDescs.push_back(trainDescs.row(perm[i]));
			testSelBuildings.push_back(trainBuildings[perm[i]]);
			testSelFloors.push_back(trainFloors[perm[i]]);
			testSelPoses.push_back(trainPoses[perm[i]]);
		}

//		// additionally noise data
//		for(int iter = 0; iter < 20; ++iter){
//			for(int d = 0; d < testSelDescs.rows; ++d){
//				bool wasOne = false;
//				for(int f = 0; f < testSelDescs.cols; f += 22){
//					int pos = 0;
//					while(pos < 22 && fabs(testSelDescs.at<float>(d, f) - 1.0) < eps){
//						++pos;
//					}
//					std::uniform_int_distribution<int> dist(0, 2);
//					int op = dist(gen);
//					// erase last 1
//					if(op == 0 && pos > 0){
//						testSelDescs.at<float>(d, f + pos - 1) = 0.0;
//					}
//					// add new 1
//					else if(op == 1 && pos < 22){
//						testSelDescs.at<float>(d, f + pos) = 1.0;
//					}
//					// do nothing
//					else{
//
//					}
//				}
//			}
//		}

		cv::Mat trainSelDescs(0, trainDescs.cols, CV_32FC1);
		std::vector<int> trainSelBuildings;
		std::vector<int> trainSelFloors;
		std::vector<cv::Point2f> trainSelPoses;
		for(int i = trainStart; i < perm.size(); ++i){
			trainSelDescs.push_back(trainDescs.row(perm[i]));
			trainSelBuildings.push_back(trainBuildings[perm[i]]);
			trainSelFloors.push_back(trainFloors[perm[i]]);
			trainSelPoses.push_back(trainPoses[perm[i]]);
		}

		cout << "trainSelDescs.rows = " << trainSelDescs.rows << endl;
		cout << "testSelDescs.rows = " << testSelDescs.rows << endl;

//		static constexpr float pzgeStart = -0.94148;
		static constexpr float pzgeStart = -0.01;
		static constexpr float pzgeStep = -0.05;
		static constexpr float pzgeStop = -4;
		static constexpr float pzgneStart = -4;
		static constexpr float pzgneStep = -0.05;
		static constexpr float pzgneStop = -10;

		static constexpr int maxThreads = 7;

		float bestScore = 0.0;
		float bestPzge, bestPzgne;
		ofstream resultsFile("validation.log");
		if(!resultsFile.good()){
			cout << "Error - couldn't open log file" << endl;
		}

		std::vector<std::vector<float>> logPzges;
		std::vector<std::vector<float>> logPzgnes;
		for(float curPzge = pzgeStart; curPzge >= pzgeStop; curPzge += pzgeStep){
			logPzges.emplace_back();
			logPzgnes.emplace_back();
			for(float curPzgne = pzgneStart; curPzgne >= pzgneStop; curPzgne += pzgneStep){
				logPzges.back().emplace_back(curPzge);
				logPzgnes.back().emplace_back(curPzgne);
			}
		}
		std::vector<std::vector<std::shared_ptr<EvaluationThread>>> threads;
		std::vector<std::vector<vector<int>>> matchedTestIdxs;
		std::vector<std::vector<int>> correct;
		std::vector<std::vector<int>> incorrect;
		std::vector<std::vector<int>> unrec;
		std::vector<std::vector<float>> results;
		for(int i = 0; i < logPzges.size(); ++i){
			matchedTestIdxs.emplace_back(logPzges[i].size());
			threads.emplace_back(logPzges[i].size());
			correct.emplace_back(logPzges[i].size());
			incorrect.emplace_back(logPzges[i].size());
			unrec.emplace_back(logPzges[i].size());
			results.emplace_back(logPzges[i].size());
		}
		int threadCnt = 0;
		for(int i = 0; i < logPzges.size(); ++i){
			for(int j = 0; j < logPzges[i].size(); ++j){

				while(threadCnt >= maxThreads){
					endThreadIfAny(threads,
									matchedTestIdxs,
									correct,
									incorrect,
									unrec,
									results,
									threadCnt,
									bestScore,
									bestPzge,
									bestPzgne,
									resultsFile);
					std::this_thread::sleep_for(chrono::milliseconds(50));
				}

				cout << "curPzge = " << logPzges[i][j] << " (" << exp(logPzges[i][j]) << ")" << endl;
				cout << "curPzgne = " << logPzgnes[i][j] << " (" << exp(logPzgnes[i][j]) << ")" << endl;

				correct.back().emplace_back();
				incorrect.back().emplace_back();
				unrec.back().emplace_back();
				results.back().emplace_back();
				threads[i][j] = shared_ptr<EvaluationThread>(new EvaluationThread(trainEnvDescs,
																trainSelDescs,
																trainSelBuildings,
																trainSelFloors,
																trainSelPoses,
																testSelDescs,
																testSelBuildings,
																testSelFloors,
																testSelPoses,
																matchedTestIdxs[i][j],
																correct[i][j],
																incorrect[i][j],
																unrec[i][j],
																exp(logPzges[i][j]),
																exp(logPzgnes[i][j]),
																clTree));
				++threadCnt;
			}
		}

		resultsFile.close();
		cout << "bestScore = " << bestScore << endl;
		cout << "bestPzge = " << bestPzge << endl;
		cout << "bestPzgne = " << bestPzgne << endl;
	}
	else{
		cout << "Recognizing places" << endl;
		cv::Mat testDescs;
		std::vector<cv::Point2f> testPoses;
		std::vector<int> testFloors;
		std::vector<int> testBuildings;
		std::vector<int> testClusters;

		loadDataFile("../res/validationData.csv",
					testDescs,
					testPoses,
					testFloors,
					testBuildings);

//		for(int d = 0; d < min(10, testDescs.rows); ++d){
//			cout << testDescs.row(d) << endl;
//		}

		preprocessDescs(testDescs);

//		for(int d = 0; d < min(10, trainDescs.rows); ++d){
//			cout << trainDescs.row(d) << endl;
//		}
//		for(int d = 0; d < min(10, testDescs.rows); ++d){
//			cout << testDescs.row(d) << endl;
//		}

		cout << "Assigning train clusters" << endl;

		assignClusters(trainPoses,
						trainFloors,
						trainClusters,
						testPoses,
						testFloors,
						testClusters,
						6.0);
		cout << (double)fsSettings["openFabMapOptions"]["PzGe"] << endl;
		cout << (double)fsSettings["openFabMapOptions"]["PzGne"] << endl;

		vector<int> matchedTestIdxs;
		int correct;
		int incorrect;
		int unrec;

		shared_ptr<EvaluationThread> evThread(new EvaluationThread(trainEnvDescs,
																	trainDescs,
																	trainBuildings,
																	trainFloors,
																	trainPoses,
																	testDescs,
																	testBuildings,
																	testFloors,
																	testPoses,
																	matchedTestIdxs,
																	correct,
																	incorrect,
																	unrec,
																	(double)fsSettings["openFabMapOptions"]["PzGe"],
																	(double)fsSettings["openFabMapOptions"]["PzGne"],
																	clTree));

		while(!evThread->hasEnded()){
			std::this_thread::sleep_for(chrono::milliseconds(50));
		}
		evThread.reset();

		ofstream matchedIdxsFile("matches.log");
		for(int i = 0; i < matchedTestIdxs.size(); ++i){
			matchedIdxsFile << matchedTestIdxs[i] << endl;
		}
	}
}
