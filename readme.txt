Hello!

To run our dialog system, please run RestaurantRecommendationSystem.py

On top of this file, you find the file paths that you will have to change so that they match to the ones on your own computer.
For the file_path_restaurants, make sure that you import restaurants_info2.csv, because that contains the extra values like crowdedness etc.

And there are 4 configurable variables:
allow_dialog_restarts: bool = True
use_delay: bool = False
output_in_caps: bool = False
use_baseline_as_classifier: bool = False

To run our data analysis and get an insight into our accuracies and the incorrect classified dialogs, run DataAnalysis.py