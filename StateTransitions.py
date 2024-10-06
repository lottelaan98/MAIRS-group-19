import pandas as pd
import Levenshtein
import random
import difflib


##################################################################################################################
#############################        CHANGE THE PATH TO MATCH YOUR COMPUTER           #############################
##################################################################################################################

keywords_2 = {
    "touristic": ["touristic"],
    "romantic": ["romantic"],
    "children": ["children", "child", "friendly"],
    "assignedseats": ["assigned seats", "reservation"],
}

keywords_1 = {
    "pricerange": ["cheap", "moderate", "expensive"],
    "area": ["north", "south", "east", "west", "centre"],
    "food": [
        "african",
        "asian oriental",
        "australasian",
        "bistro",
        "british",
        "catalan",
        "chinese",
        "cuban",
        "dutch",
        "english",
        "european",
        "french",
        "fusion",
        "gastropub",
        "indian",
        "international",
        "italian",
        "jamaican",
        "japanese",
        "korean",
        "lebanese",
        "mediterranean",
        "modern european",
        "moroccan",
        "north american",
        "persian",
        "polynesian",
        "portuguese",
        "romanian",
        "seafood",
        "spanish",
        "steakhouse",
        "swedish",
        "swiss",
        "thai",
        "traditional",
        "turkish",
        "tuscan",
        "vietnamese",
    ],
}


class Restaurant:
    def __init__(
        self,
        name,
        area,
        pricerange,
        food,
        address,
        postcode,
        phone,
        food_quality,
        crowdedness,
        length_of_stay,
    ):
        self.name = name
        self.area = area
        self.pricerange = pricerange
        self.food = food
        self.address = address
        self.postcode = postcode
        self.phone = phone
        self.food_quality = food_quality
        self.crowdedness = crowdedness
        self.length_of_stay = length_of_stay


class Helpers:

    @staticmethod
    def communicate_found_restaurant(state) -> str:
        """
        First finds the restaurants that meet the user preferences, then filters these restaurants based on the additional requirements.
        If no restaurant is found, it moves to the InformThatThereIsNoRestaurant and returns a system utterance that tells that.
        If a restaurant is found it moves into the GiveRestaurantRecommendation state and tells the user which restaurant is found and why.
        """
        # First find the restaurants based on the user preferences
        Helpers.find_restaurants1(state)

        # Then filter these found restaurants based on the additional requirements
        state.filtered_restaurants = Helpers.apply_rules(
            state.found_restaurants1, state.additional_requirements
        )

        # If no restaurant is found:
        if len(state.found_restaurants1) == 0:
            state.current_state = "InformThatThereIsNoRestaurant"
            return Helpers.tell_no_restaurant_found(state)

        # Select a restaurant and give this recommendation to the user
        state.currently_selected_restaurant = state.found_restaurants1[0]
        state.found_restaurant = state.currently_selected_restaurant
        state.current_state = "GiveRestaurantRecommendation"
        return Helpers.sell_restaurant(state)

    @staticmethod
    def find_restaurants1(state):
        """
        Looks for restaurants that match the user preferences.
        It only filters on the user_preferences. NOT ON ADDITIONAL REQUIREMENTS.
        Does not return anything. Only changes state.found_restaurants1

        If there is a restaurant found, move to GiveRestaurantRecommendation and return a string with the recommendation.
        If not found, move to InformThatThereIsNoRestaurant and return string that asks user to change requirements.
        """
        state.found_restaurants1 = []
        # Look in the CSV to find for any restaurants that may meet the criteria
        data_restaurants = pd.read_csv(state.file_path_restaurants, sep=";")
        filtered_df = data_restaurants

        # Loop through each criterion and apply the filter
        for key, value in state.user_preferences.items():
            if value == "any":
                continue
            filtered_df = filtered_df[filtered_df[key] == value]

        # Check if there are any matching restaurants
        if not filtered_df.empty:
            for _, row in filtered_df.iterrows():
                restaurant = Restaurant(
                    name=row["restaurantname"],
                    area=row["area"],
                    pricerange=row["pricerange"],
                    food=row["food"],
                    address=row["addr"],
                    postcode=row["postcode"],
                    phone=row["phone"],
                    food_quality=row["food_quality"],
                    crowdedness=row["crowdedness"],
                    length_of_stay=row["length_of_stay"],
                )
                state.found_restaurants1.append(restaurant)

    @staticmethod
    def apply_rules(
        possible_restaurants: list[Restaurant], additional_requirements: dict
    ):
        """
        Input is a list of possible restaurants and additional requirements.
        Output is a list of restaurants that meet the aditional requirements.
        """
        filtered_restaurants: list[Restaurant] = possible_restaurants

        # Filter for touristic restaurants: cheap, good or excellent food quality, food type not romanian
        if additional_requirements["touristic"] == "touristic":
            filtered_restaurants = [
                restaurant
                for restaurant in filtered_restaurants
                if restaurant.pricerange == "cheap"
                and restaurant.food_quality != "normal"
                and restaurant.food != "romanian"
            ]
        # Filter for not touristic restaurants: not cheap and good or excellent food quality
        else:
            filtered_restaurants = [
                restaurant
                for restaurant in filtered_restaurants
                if not (
                    restaurant.pricerange == "cheap"
                    and restaurant.food_quality in ["good", "excellent"]
                )
            ]

        # Filter for assigned seats: restaurants that are busy
        if additional_requirements["assignedseats"] == "assignedseats":
            filtered_restaurants = [
                restaurant
                for restaurant in filtered_restaurants
                if restaurant.crowdedness == "busy"
            ]

        # Filter for child-friendly: restaurants where the stay is not long
        if additional_requirements["children"] == "children":
            filtered_restaurants = [
                restaurant
                for restaurant in filtered_restaurants
                if restaurant.length_of_stay != "long"
            ]

        # Filter for romantic: non-busy restaurants and where the stay is long
        if additional_requirements["romantic"] == "romantic":
            filtered_restaurants = [
                restaurant
                for restaurant in filtered_restaurants
                if restaurant.crowdedness != "busy"
                and restaurant.length_of_stay == "long"
            ]

        return filtered_restaurants

    @staticmethod
    def ask_for_missing_info1(state) -> str:
        """
        State becomes/stays AskForMissingInfo.
        Returns the system utterance where it asks for the next preference that is still missing.
        """
        state.current_state = "AskForMissingInfo"
        if state.still_needed_info[0] == "area":
            return "What part of town do you have in mind?"
        if state.still_needed_info[0] == "food":
            return "What kind of food would you like?"
        if state.still_needed_info[0] == "pricerange":
            return "Would you like something in the cheap, moderate, or expensive price range?"
        raise ValueError(
            "Something went wrong in ask_for_missing info. Still_needed_info = ",
            state.still_needed_info,
        )

    @staticmethod
    def ask_for_additional_requirements(state) -> str:
        """
        State becomes/stays AskForMissingInfo.
        Returns the system utterance where it asks for the next preference that is still missing.
        """
        state.current_state = "AskForAdditionalRequirements"
        if all(value == "" for value in state.additional_requirements.values()):
            return "Are there any additional preferences you'd like to specify such as romantic atmosphere, requiring a reservation, or being child-friendly?"
        # Elif to handle partially missing values
        elif any(value == "" for value in state.additional_requirements.values()):
            missing_requirements = [
                key
                for key, value in state.additional_requirements.items()
                if value == ""
            ]
            missing_text = ", ".join(missing_requirements)
            return f"Are there any additional preferences you'd like to specify for: {missing_text}?"
        # When the user did not confirm the additional requirements:
        else:
            return "Which additional preferences would you like to specify such as romantic atmosphere, requiring a reservation, or being child-friendly?"

    @staticmethod
    def tell_no_restaurant_found(state):
        """
        Returns a system utterance that tells the user that no restaurant is found that matches his preferences.
        For example:
        'Sorry, I couldn't find a cheap italian restaurant in the north.'
        'Sorry, I couldn't find a cheap italian restaurant in any area.'

        """
        state.current_state = "InformThatThereIsNoRestaurant"

        # Initialiseer de strings
        price_string = ""
        food_string = ""
        area_string = ""

        # Loop door de criteria
        for key, value in state.user_preferences.items():
            if key == "pricerange":
                price_string = value if value != "any" else ""
            elif key == "food":
                food_string = value if value != "any" else ""
            elif key == "area":
                area_string = f"in the {value}" if value != "any" else "in any area"

        return f"Sorry, I couldn't find a {price_string} {food_string} restaurant {area_string}. Please change your requirements."

    @staticmethod
    def sell_restaurant(state):
        """
        Returns a system utterance where the found restaurant is recommended.
        For example:
        I recommend cotto in the centre, serving british cuisine, with moderate prices. Let me know if you need the post code, address or phone number.
        """

        def create_second_part():
            string = ""

            for key, value in state.additional_requirements.items():
                if value != "any":
                    if key == "children":
                        string = (
                            string
                            + " This restaurant is child-friendly because you are served quickly. "
                        )
                    elif key == "touristic":
                        string = (
                            string
                            + " It is a touristic hot-spot due to the cheap prices and the good food. "
                        )
                    elif key == "not touristic":
                        string = (
                            string
                            + " This is not a touristic place, since the food quality is very average. "
                        )
                    elif key == "assigned seats":
                        string = (
                            string
                            + " The waiter will assign you a seat, since it is a very busy restaurant. "
                        )
                    elif key == "romantic":
                        string = (
                            string
                            + " The restaurant is romantic because it allows you to stay for a long time. "
                        )
            return string

        first_part = f"I recommend {state.found_restaurant.name} in the {state.found_restaurant.area} area, serving {state.found_restaurant.food} cuisine, with {state.found_restaurant.pricerange} prices."
        second_part = create_second_part()
        third_part = "Let me know if you need the post code, address or phone number."

        return first_part + second_part + third_part

    @staticmethod
    def ask_for_confirmation1(state) -> str:
        """
        Moves the state to 'AskForConfirmation1' and returns the corresponding system utterance.
        """
        state.current_state = "AskForConfirmation1"

        if len(state.user_preferences) != 3:
            raise ValueError(
                "User_preferences doesn't have 3 keys: \n ", state.user_preferences
            )

        food = state.user_preferences["food"]
        area = state.user_preferences["area"]
        pricerange = state.user_preferences["pricerange"]

        vowels = "aeiou"

        # Preferred food type
        if food == "any":
            food_text = "any"
        else:
            article = "an" if food[0] in vowels else "a"
            food_text = f"{article} {food}"

        # Preferred area
        if area == "any":
            area_text = "any area"
        else:
            area_text = f"the {area} of town"

        # Preferred price range
        if pricerange == "any":
            pricerange_text = "and you don't care about the price range"
        else:
            pricerange_text = f"in the {pricerange} price range"

        sentence = f"Let me confirm, you are looking for {food_text} restaurant in {area_text} {pricerange_text}, right?"

        return sentence

    @staticmethod
    def ask_for_confirmation2(state):
        state.current_state = "AskForConfirmation2"
        # List to store preferences
        preferencesss = []

        # Check each preference and append the non-empty ones
        if state.additional_requirements["touristic"] != "any":
            preferencesss.append(f" Touristic")
        if state.additional_requirements["romantic"] != "any":
            preferencesss.append(f" Romantic")
        if state.additional_requirements["children"] != "any":
            preferencesss.append(f" Children allowed")
        if state.additional_requirements["assignedseats"] != "any":
            preferencesss.append(
                f"with {state.additional_requirements['assignedseats']}"
            )

        # Join preferences into a sentence
        if preferencesss:
            confirmation_message = f"To clarify, you prefer a restaurant with these qualities: {', '.join(preferencesss)}."
        else:
            confirmation_message = (
                "Let me confirm, you have not specified any preferences."
            )

        return confirmation_message

    def detect_any(key, last_system_utterance) -> bool:
        """
        It can happen that the system asks: "What part of town do you have in mind?"
        And the user answers: "any"
        Then we need to find out what this "any" was about.

        """
        if key == "area" and "town" in last_system_utterance:
            return True
        elif key == "food" and "kind of food" in last_system_utterance:
            return True
        elif (
            key == "pricerange"
            and "cheap, moderate, or expensive price range" in last_system_utterance
        ):
            return True
        else:
            return False

    def extract_additional_requirements(state, user_input):
        """
        Extract preferences (romantic, children, assignedseats, touristic) from user_input and put in state.additional_requirements
        """
        keys = ["touristic", "romantic", "children", "assignedseats"]
        result = {key: "any" for key in keys}
        user_input = user_input.lower()

        keywords_2 = {
            "touristic": ["touristic"],
            "romantic": ["romantic"],
            "children": ["children"],
            "assignedseats": ["assigned seats", "reservation"],
        }

        negations = ["no", "not", "don't", "do not", "without", "none"]

        input_words = user_input.split()
        for key, words in keywords_2.items():
            for word in words:
                if word in user_input:
                    word_idx = user_input.find(word)
                    negated = any(neg in user_input[:word_idx] for neg in negations)

                    if negated:
                        result[key] = f"not {word}"
                    else:
                        result[key] = word
                    break
        for key, value in result.items():
            if value == "any":
                for word in keywords_2[key]:
                    matches = difflib.get_close_matches(word, input_words, cutoff=0.8)
                    if matches:
                        result[key] = word
                        break

        if result["assignedseats"] in ["assigned seats", "reservation"]:
            result["assignedseats"] = "assignedseats"

        state.additional_requirements = result

    @staticmethod
    def extract_preferences(state, user_input, overwrite):
        """
        Extract preferences (food, area, pricerange) from user_input and put in state.user_preferences
        """
        preference_extracted = False
        # Keyword matching: Check if there is a preference expressed in the user input
        for key, words in keywords_1.items():
            for word in words:
                if word in user_input:
                    preference_extracted = True
                    # Check for ambiguity
                    if key in state.user_preferences and not overwrite:
                        # Remove this key from user preferences and add to the ambiguity dictionary and to the still needed info.
                        state.ambiguity[key] = [state.user_preferences[key], word]
                        state.user_preferences.pop(key)
                        state.still_needed_info.append(key)
                    else:
                        state.user_preferences[key] = word
                        if key in state.still_needed_info:
                            state.still_needed_info.remove(key)

        # Handle 'any' as a wildcard
        if any(
            keyword in user_input
            for keyword in [
                "any",
                "dont care",
                "don't care",
                "do not care",
                "doesnt matter",
                "doesn't matter",
            ]
        ):
            for key in keywords_1:
                if key in user_input or Helpers.detect_any(
                    key, state.last_system_utterance
                ):

                    preference_extracted = True
                    # Check for ambiguity
                    if key in state.user_preferences and not overwrite:
                        # Remove this key from user preferences and add to the ambiguity dictionary.
                        state.ambiguity[key] = [state.user_preferences[key], "any"]
                        state.user_preferences.pop(key)
                        state.still_needed_info.append(key)
                    else:
                        state.user_preferences[key] = "any"
                        if key in state.still_needed_info:
                            state.still_needed_info.remove(key)

        # Use Levenshtein algorithm if no matches found
        if not preference_extracted:
            result = Helpers.perform_levenshtein(user_input)
            if result is not None:
                key, word = result
                state.user_preferences[key] = word
                if key in state.still_needed_info:
                    state.still_needed_info.remove(key)

    @staticmethod
    def perform_levenshtein(user_input):
        """
        Performs Levenshtein algorithm on a certain user input. Returns
        """
        for key, words in keywords_1.items():
            for word in words:
                if any(
                    Levenshtein.ratio(word, token) > 0.8 for token in user_input.split()
                ):
                    return (key, word)
        return None

    @staticmethod
    def ask_user_for_clarification(state):
        """
        Looks into the state.ambiguity dictionary, and asks the user for which `key` he would like which value.
        For example:
        'For the food preference, would you like british or romanian?'
        'For the pricerange preference, would you like cheap or expensive?'
        """
        key = list(state.ambiguity.keys())[0]
        possible_values = state.ambiguity[key]
        return f"For the {key} preference, would you like {possible_values[0]} or {possible_values[1]}?"

    @staticmethod
    def fix_ambiguity(state, user_input) -> str:
        """
        Checks the user_input to see which value for a specific preference (key) the user chooses.
        It adds the correct preference to the state.user_preferences dictionary and removes this key from the ambiguity dictionary.
        """
        key = list(state.ambiguity.keys())[0]
        possible_values = state.ambiguity[key]

        for value in possible_values:
            if value in user_input:
                state.still_needed_info.remove(key)
                state.user_preferences[key] = value
                state.ambiguity.pop(key)
                return f"Your choice for {key} is now set to: {value}. "
        return "I'm sorry, I failed to adjust your choice. "

    @staticmethod
    def provide_all_contact_info(restaurant) -> str:
        return f"The restaurant {restaurant.name} is on {restaurant.address} with post code {restaurant.postcode}. This is their phone number: {restaurant.phone}"


class Dialog_Acts:

    def ack(self, state):
        """
        When in ProvideContactInformation: Provide all contact information
        When in GiveRestaurantRecommendation: If there is one, give another recommendation based on the state.found_restaurants list.
        """
        system_utterance = state.last_system_utterance
        if state.current_state == "ProvideContactInformation":
            system_utterance = Helpers.provide_all_contact_info(
                state.found_restaurants1[0]
            )
        elif state.current_state == "GiveRestaurantRecommendation":
            system_utterance = self.reqmore(state)
        else:
            system_utterance = "How else can I help you?"

        return system_utterance

    def affirm(self, state):
        system_utterance = state.last_system_utterance
        if state.current_state == "AskForConfirmation1":
            system_utterance = Helpers.ask_for_additional_requirements(state)
        elif state.current_state == "AskForConfirmation2":
            system_utterance = Helpers.communicate_found_restaurant(state)

        elif state.current_state == "AskUserForClarification":
            # If the user confirmed the last system utterance
            words = state.last_system_utterance.split()
            if "town" in words[-1]:
                state.user_preferences["area"] = state.last_system_utterance.split()[11]
                if "area" in state.still_needed_info:
                    state.still_needed_info.remove("area")
                system_utterance = (
                    f'Okay. The area is now set to {state.user_preferences["area"]}. '
                )
            elif "range" in words[-1]:
                state.user_preferences["pricerange"] = (
                    state.last_system_utterance.split()[11]
                )
                if "pricerange" in state.still_needed_info:
                    state.still_needed_info.remove("pricerange")
                system_utterance = f'Okay. The price range is now set to {state.user_preferences["pricerange"]}. '
            elif "food" in words[-1]:
                state.user_preferences["food"] = state.last_system_utterance.split()[11]
                if "food" in state.still_needed_info:
                    state.still_needed_info.remove("food")
                system_utterance = f'Okay. Your food preference is now set to {state.user_preferences["food"]}. '

            extra_text = ""

            if state.still_needed_info:
                extra_text = Helpers.ask_for_missing_info1(state)
            elif state.current_state == "AskForConfirmation2":
                extra_text = Helpers.ask_for_confirmation2(state)
            else:
                extra_text = Helpers.ask_for_confirmation1(state)

            system_utterance += extra_text

        elif state.current_state == "InformThatThereIsNoRestaurant":
            system_utterance = "Please provide your new preferences."

        return system_utterance

    def bye(self, state):
        state.current_state = "End"
        return "Goodbye! Enjoy your meal!"

    def confirm(self, state, user_input) -> str:
        """
        User input is a question for the user where he wants to confirm some information.
        Return a string with the answer.
        """
        if state.found_restaurants1 == []:
            raise ValueError(
                "There are no found restaurants when calling the confirm() function"
            )

        # Keyword matching: Check if there is a keyword the user input
        for key, words in keywords_1.items():
            for word in words:
                if word in user_input:
                    restaurant = state.currently_selected_restaurant
                    if key == "pricerange":
                        return f"{restaurant.name} is a nice place in the {restaurant.area} of town."
                    elif key == "food":
                        return f"{restaurant.name} is a nice place serving {restaurant.food} food."
                    else:
                        return f"{restaurant.name} is a nice place in the {restaurant.pricerange} price range."

        return "Could you please repeat that?"

    def deny(self, state, user_input) -> str:
        """
        Removes the preference that the user denies or removes all preferences when not specified which one.
        Then asks for the missing preferences.
        """
        state.current_state = "AskForMissingInfo"

        # Keyword matching: Check if there is a keyword in the user input
        for key, words in keywords_1.items():
            for word in words:
                if word in user_input:
                    # Remove this key from the user_preferences and ask for new one
                    del state.user_preferences[key]
                    state.still_needed_info.append(key)
                    return f"Okay. What is your preference for {key}?"

        # If there was no key word in here, remove all preferences.
        state.still_needed_info = ["area", "food", "pricerange"]
        state.user_preferences = {}
        return "Let us try again. Could you provide me more information about your preferred area, price range and food type?"

    def hello(self, state):
        if state.current_state == "Welcome":
            return "Could you provide me more information about your preferred area, price range and food type?"
        else:
            return state.last_system_utterance

    def inform(self, state, user_input) -> str:
        """
        Extracts the preferences form the user utterance.
        If the system needs more information, it will ask for missing info.
        If not, it wil go to the AskForConfirmation state.
        Return system utterance
        """
        # When the user just answered the question where the system asked for clarification, this ambiguity needs to be solved.
        if state.current_state == "AskUserForClarification":
            system_utterance1 = Helpers.fix_ambiguity(state, user_input)
            # If the system has all the info it needs, then ask for confirmation
            if not state.still_needed_info:
                return Helpers.ask_for_confirmation1(state)
            else:
                state.current_state = "AskForMissingInfo1"
                system_utterance2 = Helpers.ask_for_missing_info1(state)
                return system_utterance1 + system_utterance2
        # Extract preferences from the dialog_act only if there is no ambiguity to be solved.
        else:
            Helpers.extract_preferences(
                state,
                user_input,
                (
                    state.current_state == "InformThatThereIsNoRestaurant"
                    or state.current_state == "Welcome"
                ),
            )

        if state.current_state == "AskForAdditionalRequirements":
            Helpers.extract_additional_requirements(state, user_input)

        # First fix the ambiguity of the user input
        if state.ambiguity != {}:
            state.current_state = "AskUserForClarification"
            system_utterance = Helpers.ask_user_for_clarification(state)

        # Find a system utterance based on the preferences that are still missing
        elif state.still_needed_info:
            # System moves to state AskForMissingInfo
            system_utterance = Helpers.ask_for_missing_info1(state)
        elif state.current_state == "AskForAdditionalRequirements":

            system_utterance = Helpers.ask_for_confirmation2(state)
        else:
            # System moves to state AskForConfirmation
            system_utterance = Helpers.ask_for_confirmation1(state)

        return system_utterance

    def negate(self, state, user_input) -> str:
        """
        If the user negated the first confirmation, change the user preferences if he stated a new preference.
        If he didn't state a preference, then start over with the user preferences.

        If the user negated the question of the additional requirements, put 'any' in the additional requirements.

        If the user negated the second confirmation, ask again.

        """
        system_utterance = state.last_system_utterance
        if (
            state.current_state == "AskForConfirmation1"
            or state.current_state == "AskForMissingInfo1"
        ):
            previous_preferences = state.user_preferences
            Helpers.extract_preferences(state, user_input, True)
            # When the user didn't state what he wants to change, remove all preferences and ask again.
            if previous_preferences == state.user_preferences:
                state.user_preferences = {}
                state.still_needed_info = ["area", "food", "pricerange"]
                system_utterance = (
                    "I'm sorry. What is your preference for food, price range and area?"
                )
            # When the user did change what he want to change, this is in the user_preferences now and you can ask for confirmation.
            else:
                system_utterance = Helpers.ask_for_confirmation1(state)
        elif state.current_state == "AskForAdditionalRequirements":
            # The user doesn't care about additional requirements.
            Helpers.extract_additional_requirements(state, "any")
            state.current_state = "AskForConfirmation2"
            system_utterance = Helpers.ask_for_confirmation2(state)
        elif state.current_state == "AskForConfirmation2":
            system_utterance = Helpers.ask_for_additional_requirements(state)

        return system_utterance

    def null(self, state, user_input):
        system_utterance = state.last_system_utterance

        words = user_input.lower().split()

        keyword_found = False
        # Loop through the keyword dictionary
        for _, keywords in keywords_2.items():
            # Check if any keyword from the dictionary is in the user's input
            if any(keyword in words for keyword in keywords):
                keyword_found = True
        if keyword_found:
            state.current_state = "AskForConfirmation2"
            Helpers.extract_additional_requirements(state, user_input)
            return Helpers.ask_for_confirmation2(state)

        if state.current_state == "AskForAdditionalRequirements":
            return Helpers.communicate_found_restaurant(state)

        if (
            state.current_state == "AskForMissingInfo1"
            or state.current_state == "Welcome"
        ):
            result = Helpers.perform_levenshtein(user_input)

            if result is not None:
                state.current_state = "AskUserForClarification"
                key, word = result
                if key == "area":
                    system_utterance = f"Did you say you are looking for a restaurant in the {word} of town?"
                elif key == "pricerange":
                    system_utterance = f"Did you say you are looking for a restaurant in the {word} price range?"
                elif key == "food":
                    system_utterance = f"Did you say you are looking for a restaurant that serves {word} food?"
            if result is None and state.current_state == "Welcome":
                # System moves to State AskForMissingInfo
                system_utterance = "I'm sorry, I did not get that. Please give your preferences for area, price range and food type. "
        elif state.current_state == "AskUserForClarification":
            Helpers.fix_ambiguity(state, user_input)
            if state.ambiguity == {}:
                # Find a system utterance based on the preferences that are still missing
                if state.still_needed_info:
                    # System moves to state AskForMissingInfo
                    system_utterance = Helpers.ask_for_missing_info1(state)
                else:
                    # System moves to state AskForConfirmation
                    system_utterance = Helpers.ask_for_confirmation1(state)
            else:
                system_utterance = Helpers.ask_user_for_clarification(state)
        else:
            system_utterance = "I am sorry. Could you please rephrase that?"

        return system_utterance

    def repeat(self, state):
        return state.last_system_utterance

    def reqalts(self, state, user_input):
        """
        When in state InformThatThereIsNoRestaurant: The user_input will be in the form of "How about...".
        Then we run find_restaurants again with the new preference.

        When in state GiveRestaurantRecommendation, the user wants another restaurant -> Remove the first
        restaurant from the restaurant list and give the new number 1.
        """

        system_utterance = state.last_system_utterance

        if state.current_state == "InformThatThereIsNoRestaurant":
            Helpers.extract_preferences(state, user_input, True)

        def detect_unlisted_keywords():
            words = user_input.split()  # Split the sentence into words

            # Iterate over each word in the user input
            for word in words:
                for category in keywords_1:
                    # If the word is found in the keywords of any category and not in preferences
                    if (
                        word in keywords_1[category]
                        and word not in state.user_preferences
                    ):
                        state.user_preferences[category] = word
                        return True

            return False

        system_utterance = state.last_system_utterance

        if detect_unlisted_keywords():
            state.found_restaurants1 = []
            system_utterance = Helpers.communicate_found_restaurant(state)

        elif state.current_state == "GiveRestaurantRecommendation":
            Helpers.extract_preferences(state, user_input, True)
            Helpers.extract_additional_requirements(state, user_input)
            # Remove the current found restaurant from the list
            del state.found_restaurants1[0]
            # and provide another restaurant
            if not state.found_restaurants1:
                state.current_state = "InformThatThereIsNoRestaurant"
                system_utterance = "Sorry, I couldn't find another restaurant that matches your preferences. Can you change your requirements?"
                words = user_input.lower().split()
                for _, keywords in keywords_2.items():
                    if any(keyword in words for keyword in keywords):
                        state.current_state = "AskForAdditionalRequirements"
            else:
                state.currently_selected_restaurant = state.found_restaurants1[0]
                system_utterance = Helpers.sell_restaurant(state)

        return system_utterance

    def reqmore(self, state):
        other_recommendations = [
            restaurant
            for restaurant in state.found_restaurants1
            if restaurant.name != state.currently_selected_restaurant.name
        ]
        if not other_recommendations:
            system_utterance = f"The restaurant {state.currently_selected_restaurant.name} is the only restaurant that meets your preferences."
        else:
            state.currently_selected_restaurant = random.choice(other_recommendations)
            system_utterance = Helpers.sell_restaurant(state)
        return system_utterance

    def request(self, state, user_input):
        def get_closest_word(user_input, target_words, threshold=1):
            for target_word in target_words:
                for word in user_input.split():
                    if Levenshtein.distance(word.lower(), target_word) <= threshold:
                        return target_word
            return None

        input_words = user_input.split()
        testtt = False
        for key, words in keywords_2.items():
            for word in words:
                if word in input_words:
                    testtt = True
        output_text = ""

        address_words = ["address"]
        post_words = ["post"]
        phone_words = ["phone"]
        if get_closest_word(user_input, address_words):
            output_text = f"The address of {state.currently_selected_restaurant.name} is on {state.currently_selected_restaurant.address}. "

        if get_closest_word(user_input, post_words):
            output_text += f"The post code of {state.currently_selected_restaurant.name} is {state.currently_selected_restaurant.postcode}. "

        if get_closest_word(user_input, phone_words):
            output_text += f"The phone number of {state.currently_selected_restaurant.name} is {state.currently_selected_restaurant.phone}."

        if testtt:
            state.current_state = "AskForMissingInfo2"
            Helpers.extract_additional_requirements(state, user_input)
        if output_text == "":
            return "Can you repeat that please?"

        return output_text

    def restart(self, state, allowed):
        """
        If it is allowed, it will reset all the values in the state to the starting values and it returns a new system utterance,
        where it mentions that the system has started over.
        """
        if allowed:
            state.current_state = "Welcome"
            state.user_preferences = {}
            state.still_needed_info = ["area", "food", "pricerange"]
            state.dialog_acts = Dialog_Acts()
            state.helpers = Helpers()
            state.last_system_utterance = "Okay. We start over. Welcome to the UU restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?"
            state.ambiguity = {}
            state.found_restaurants1 = []
            state.filtered_restaurants = []
            state.currently_selected_restaurant = None
            state.additional_requirements = {
                "touristic": "",
                "romantic": "",
                "children": "",
                "assignedseats": "",
            }
            return state.last_system_utterance
        else:
            return "I'm sorry. You are not allowed to restart the dialog."

    def thankyou(self, state, user_input):
        if "bye" in user_input:
            state.current_state = "End"
            return "You're welcome. Good bye."
        else:
            return "You're welcome."


class State:
    def __init__(self, file_path_restaurants):
        self.current_state = "Welcome"
        self.user_preferences = {}
        self.still_needed_info: list[str] = ["area", "food", "pricerange"]
        self.dialog_acts = Dialog_Acts()
        self.helpers = Helpers()
        self.last_system_utterance = ""
        self.ambiguity = {}
        self.found_restaurants1 = []
        self.filtered_restaurants = []
        self.currently_selected_restaurant: Restaurant = None
        self.file_path_restaurants = file_path_restaurants
        self.additional_requirements = {
            "touristic": "",
            "romantic": "",
            "children": "",
            "assignedseats": "",
        }
