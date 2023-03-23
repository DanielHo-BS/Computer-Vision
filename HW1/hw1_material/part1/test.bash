python main.py --threshold 3.0
mkdir out/test1
mv out/*.png out/test1

mkdir out/test2
python main.py --threshold 1.0 --image_path ./testdata/2.png
mkdir out/test2/thershold-1
mv out/*.png out/test2/thershold-1
python main.py --threshold 2.0 --image_path ./testdata/2.png
mkdir out/test2/thershold-2
mv out/*.png out/test2/thershold-2
python main.py --threshold 3.0 --image_path ./testdata/2.png
mkdir out/test2/thershold-3
mv out/*.png out/test2/thershold-3