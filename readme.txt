Hello!

We have two files that you can run: The RestaurantRecommendationSystem.py and the DataAnalysis.py file.

To run our dialog system, please run RestaurantRecommendationSystem.py

On top of this file, you find the file paths that you will have to change so that they match to the ones on your own computer.
For the file_path_restaurants, make sure that you import restaurants_info2.csv, because that contains the extra values like crowdedness etc.

And there are 4 configurable variables:
allow_dialog_restarts: bool = True
use_delay: bool = False
output_in_caps: bool = False
use_baseline_as_classifier: bool = False

You can configure these as you want.



To run our data analysis and get an insight into our accuracies and the error analysis et cetera, run DataAnalysis.py. However, before running, also change this file path to match the one on your computer.
Please note that running DataAnalysis.py will create files in your repository that contain the incorrect classified utterances.


Kind regards,
Group 19