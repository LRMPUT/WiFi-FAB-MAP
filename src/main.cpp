
#include <list>
#include <map>
#include <fstream>
#include <chrono>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>

#include "../include/openfabmap.hpp"

using namespace std;
using namespace cv;

static constexpr int descSize = 520;
static constexpr float matchProbThresh = 0.0;
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

void preprocessDescs(cv::Mat& descs){
	static constexpr int minVal = -110;
	static constexpr int maxVal = 0;
	static constexpr int stepVal = 10;

	descs -= Scalar(minVal);
	Mat notFoundMask = (descs == 100 - minVal);
	descs.setTo(Scalar(0), notFoundMask);
	int binsPerDesc = ((maxVal - minVal)/stepVal);
	Mat extDescs(descs.rows, binsPerDesc * descs.cols, CV_32FC1, Scalar(0));
	for(int d = 0; d < descs.rows; ++d){
		for(int v = 0; v < descs.cols; ++v){
//			cout << "v = " << v << endl;
			if(fabs(descs.at<float>(d, v)) > eps){
				float val = descs.at<float>(d, v);
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
	
	loadDataFile("../res/trainingData.csv",
				trainDescs,
				trainPoses,
				trainFloors,
				trainBuildings);

//	for(int d = 0; d < min(10, trainDescs.rows); ++d){
//		cout << trainDescs.row(d) << endl;
//	}

	preprocessDescs(trainDescs);

//	for(int d = 0; d < min(10, trainDescs.rows); ++d){
//		cout << trainDescs.row(d) << endl;
//	}


	cv::Mat clTree;
	if((int)fsSettings["ChowLiuOptions"]["Train"] > 0){
		cout << "Learning Chow-Liu Tree" << endl;
		of2::ChowLiuTree tree;
		tree.add(trainDescs);
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

//	cout << (double)fsSettings["openFabMapOptions"]["PzGe"] << endl;
//	cout << (double)fsSettings["openFabMapOptions"]["PzGne"] << endl;

	shared_ptr<of2::FabMap> fabmap(new of2::FabMap2(clTree,
												(double)fsSettings["openFabMapOptions"]["PzGe"],
												(double)fsSettings["openFabMapOptions"]["PzGne"],
												of2::FabMap::SAMPLED | of2::FabMap::CHOW_LIU));

	cout << "Adding train data" << endl;
	fabmap->addTraining(trainDescs);

	cout << "Adding places database" << endl;
	fabmap->add(trainDescs);

	cout << "Recognizing places" << endl;
	cv::Mat testDescs;
	std::vector<cv::Point2f> testPoses;
	std::vector<int> testFloors;
	std::vector<int> testBuildings;

	loadDataFile("../res/validationData.csv",
				testDescs,
				testPoses,
				testFloors,
				testBuildings);

	preprocessDescs(testDescs);

//	for(int d = 0; d < min(10, testDescs.rows); ++d){
//		cout << testDescs.row(d) << endl;
//	}


	int correct = 0;
	int incorrect = 0;
	int unrec = 0;
	int all = 0;
	for(int d = 0; d < testDescs.rows; ++d){
		std::vector<of2::IMatch> matches;
		fabmap->compare(testDescs.row(d), matches, false);
		double bestMatchProb = 0.0;
		int bestMatchTestIdx = -1;
		std::cout << "matches.size() = " << matches.size() << std::endl;
		for(std::vector<of2::IMatch>::iterator it = matches.begin(); it != matches.end(); ++it){
//			cout << "it->match = " << it->match << endl;
			if(it->match > matchProbThresh){
				if(bestMatchProb < it->match){
					bestMatchProb = it->match;
					bestMatchTestIdx = it->imgIdx;
				}
			}
		}
		std::cout << "Matched with " << bestMatchTestIdx << ", prob = " << bestMatchProb << std::endl;
		if(bestMatchTestIdx >= 0){
			if(trainBuildings[bestMatchTestIdx] == testBuildings[d] &&
				trainFloors[bestMatchTestIdx] == testFloors[d])
			{
				++correct;
			}
			else{
				++incorrect;
			}
		}
		else{
			++unrec;
		}
	}
	cout << "correct = " << correct << endl;
	cout << "incorrect = " << incorrect << endl;
	cout << "unrec = " << unrec << endl;
	cout << "success rate = " << (double)correct / (correct + incorrect + unrec) << endl;
}
