mkdir dependencies
cd dependencies
git clone https://github.com/shmsw25/FActScore.git
git clone https://github.com/sylinrl/TruthfulQA.git
cd scripts
mv * ..
cd ..
cd src
mv * ..
cd ..
cd dependencies
mv FActScore ..
cd ..
cd data/processed_retrieval_data
mv * ../..
cd ../..
rm -r scripts
rm -r src
./setup.sh
