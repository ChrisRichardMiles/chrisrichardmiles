echo "Generating features"
python generate_features_p13.py
echo "Training best dart model"
python dart_model.py
echo "Finished"