cd scripts
mv * ..
cd ..
cd src
mv * ..
cd ..
cd Dependencies
mv FActScore ..
cd ..
cd data/processed_retrieval_data
mv * ../..
cd ../..
rm -r scripts
rm -r src
./setup.sh
