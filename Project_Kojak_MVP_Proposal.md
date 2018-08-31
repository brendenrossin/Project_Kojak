## Project Proposal

The minimum viable product that I'm hoping to present on would be a time-series model to predict weight and biometric activity fluctuations based on information obtained from my Fitbit app (steps, activity, sleep) through its API and my MyFitnessPal account which tracks calorie intake, macronutrient information, and weight. In addition, I'd like to implement a meal recommendation system by using recipes taken from FoodNetwork.com (or through Yummly API) which would satisfy a certain calorie goal, while maximizing macronutrient (protein, fat, carbs) targets.

I'm planning on implementing an Flask app which will display today's current calorie intake, the breakdown on macronutrients eaten so far, the calories remaining based on a specified calorie goal, and activity trackers. The app will then have a drop down option for type of cuisine the user is looking to have for their next meal, and then have recommendations based off the remaining calorie count and how well they satisfy the macronutrient goal as well.

The ultimate goal is to have a projection on the Flask app for how the user's future caloric intake will affect their weight fluctuation in the future, within some confidence interval.

As a "like-to-have", I would hope to implement a "Cravings Predictor", which could be implemented by looking at previous food entries to predict what type of food I might be craving given certain features such as:
* time of day
* amount of activity
* day of week
* recency of that food being eaten.

Another "like-to-have" would be functionality in the app to alert users of abnormal health monitoring issues such as:
    1. Abnormal sleeping patterns
    2. Abnormal nutrient imbalance
    3. Vitamin deficiency
    4. Below threshold of “acceptable” activity

If I could then incorporate the cravings index to the recommendations from the Food Network/Yummly recipes, I could show food recommendations with no user input and the impact that meal would have on future weight and biometric statistics.

Last, I hope to build the app to update "real-time", if possible. Update with new data weekly?

### Dataset

My dataset is going to come from my Fitbit data, which I've already been able to obtain using the API. I have data going back from December 2016, where I have ~80% of days logged, and I'm hoping to impute the remaining days based on the day of the week and the recent trends from that time. I have also obtained all of my meals that I've logged and my weight (logged daily) over the past 2 years from MyFitnessPal. I've interpolated my weights linearly already which I have confidence in, and I'm going to impute my missing meals similarly to how I described imputing my steps for Fitbit.

Lastly, I'm going to try and gain access to the Yummly API to gather recipes and their nutritional information in order to make recommendations on my Flask app. If I can't, I'm planning on getting data from FoodNetwork because they have recipe and nutritional information as well.

### Domain

The domain is me, which is my favorite domain, and a domain with which I'm very familiar. I have a very good understanding of the general impact my activity levels and caloric intake already have on my weight, so I can't foresee running into any issues there.

### Known unknowns

I'm very unfamiliar with time-series models and the outputs I can obtain from them, or how ambitious this project is. If this seems too "easy" of a project I could potentially work on creating a workout/running route recommender as well, but that seems ambitious and could even be its own project.
