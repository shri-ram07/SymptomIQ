from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import os
import google.generativeai as genai


app = Flask(__name__)

# Load the TensorFlow model
model = tf.saved_model.load("assets/saved_models")

# Original list of symptoms (technical names)
ORIGINAL_SYMPTOMS = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills",
    "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting",
    "burning_micturition", "spotting_urination", "fatigue", "weight_gain", "anxiety",
    "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", "lethargy",
    "patches_in_throat", "irregular_sugar_level", "cough", "high_fever", "sunken_eyes",
    "breathlessness", "sweating", "dehydration", "indigestion", "headache", "yellowish_skin",
    "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain",
    "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
    "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach",
    "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm", "throat_irritation",
    "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "chest_pain",
    "weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements", "pain_in_anal_region",
    "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising",
    "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid",
    "brittle_nails", "swollen_extremities", "excessive_hunger", "extra_marital_contacts",
    "drying_and_tingling_lips", "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness",
    "stiff_neck", "swelling_joints", "movement_stiffness", "spinning_movements", "loss_of_balance",
    "unsteadiness", "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort",
    "foul_smell_of urine", "continuous_feel_of_urine", "passage_of_gases", "internal_itching",
    "toxic_look_(typhos)", "depression", "irritability", "muscle_pain", "altered_sensorium",
    "red_spots_over_body", "belly_pain", "abnormal_menstruation", "dischromic_patches",
    "watering_from_eyes", "increased_appetite", "polyuria", "family_history", "mucoid_sputum",
    "rusty_sputum", "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion",
    "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen",
    "history_of_alcohol_consumption", "fluid_overload", "blood_in_sputum", "prominent_veins_on_calf",
    "palpitations", "painful_walking", "pus_filled_pimples", "blackheads", "scurring", "skin_peeling",
    "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", "blister",
    "red_sore_around_nose", "yellow_crust_ooze"
]

# User-friendly symptom names and descriptions
SYMPTOM_DETAILS = {
    "itching": {"name": "Itchy Skin", "description": "Persistent itching on the skin."},
    "skin_rash": {"name": "Skin Rash", "description": "Red or swollen patches on the skin."},
    "nodal_skin_eruptions": {"name": "Skin Bumps", "description": "Small raised areas on the skin."},
    "continuous_sneezing": {"name": "Frequent Sneezing", "description": "Non-stop sneezing episodes."},
    "shivering": {"name": "Shivering", "description": "Involuntary shaking due to cold or fever."},
    "chills": {"name": "Chills", "description": "Feeling cold without an actual drop in body temperature."},
    "joint_pain": {"name": "Joint Pain", "description": "Ache or discomfort in the joints."},
    "stomach_pain": {"name": "Stomach Ache", "description": "Pain or discomfort in the abdominal area."},
    "acidity": {"name": "Acid Reflux", "description": "Burning sensation in the chest due to stomach acid."},
    "ulcers_on_tongue": {"name": "Mouth Ulcers", "description": "Painful sores on the tongue or mouth."},
    "muscle_wasting": {"name": "Muscle Weakness", "description": "Loss of muscle strength and mass."},
    "vomiting": {"name": "Vomiting", "description": "Expelling contents of the stomach through the mouth."},
    "burning_micturition": {"name": "Burning Urination", "description": "Pain or burning sensation while urinating."},
    "spotting_urination": {"name": "Frequent Urination", "description": "Need to urinate more often than usual."},
    "fatigue": {"name": "Fatigue", "description": "Extreme tiredness or lack of energy."},
    "weight_gain": {"name": "Weight Gain", "description": "Unexplained increase in body weight."},
    "anxiety": {"name": "Anxiety", "description": "Feelings of worry or fear."},
    "cold_hands_and_feets": {"name": "Cold Hands and Feet", "description": "Constantly cold extremities."},
    "mood_swings": {"name": "Mood Swings", "description": "Sudden changes in mood."},
    "weight_loss": {"name": "Weight Loss", "description": "Unexplained decrease in body weight."},
    "restlessness": {"name": "Restlessness", "description": "Inability to relax or stay still."},
    "lethargy": {"name": "Lethargy", "description": "Lack of energy or enthusiasm."},
    "patches_in_throat": {"name": "Throat Patches", "description": "White or red patches in the throat."},
    "irregular_sugar_level": {"name": "Irregular Blood Sugar", "description": "Fluctuating blood sugar levels."},
    "cough": {"name": "Cough", "description": "Persistent coughing."},
    "high_fever": {"name": "High Fever", "description": "Body temperature above 101°F (38.3°C)."},
    "sunken_eyes": {"name": "Sunken Eyes", "description": "Eyes appearing hollow or deeply set."},
    "breathlessness": {"name": "Shortness of Breath", "description": "Difficulty breathing."},
    "sweating": {"name": "Excessive Sweating", "description": "Unusual or excessive sweating."},
    "dehydration": {"name": "Dehydration", "description": "Lack of adequate water in the body."},
    "indigestion": {"name": "Indigestion", "description": "Discomfort or pain in the upper abdomen."},
    "headache": {"name": "Headache", "description": "Pain in the head or neck area."},
    "yellowish_skin": {"name": "Yellow Skin", "description": "Skin appearing yellowish due to jaundice."},
    "dark_urine": {"name": "Dark Urine", "description": "Urine appearing darker than usual."},
    "nausea": {"name": "Nausea", "description": "Feeling of sickness or urge to vomit."},
    "loss_of_appetite": {"name": "Loss of Appetite", "description": "Reduced desire to eat."},
    "pain_behind_the_eyes": {"name": "Eye Pain", "description": "Pain behind or around the eyes."},
    "back_pain": {"name": "Back Pain", "description": "Ache or discomfort in the back."},
    "constipation": {"name": "Constipation", "description": "Difficulty passing stools."},
    "abdominal_pain": {"name": "Abdominal Pain", "description": "Pain in the stomach area."},
    "diarrhoea": {"name": "Diarrhea", "description": "Frequent loose or watery stools."},
    "mild_fever": {"name": "Mild Fever", "description": "Slightly elevated body temperature."},
    "yellow_urine": {"name": "Yellow Urine", "description": "Urine appearing unusually yellow."},
    "yellowing_of_eyes": {"name": "Yellow Eyes", "description": "Whites of the eyes turning yellow."},
    "acute_liver_failure": {"name": "Liver Failure", "description": "Sudden loss of liver function."},
    "fluid_overload": {"name": "Fluid Retention", "description": "Swelling due to excess fluid in the body."},
    "swelling_of_stomach": {"name": "Stomach Swelling", "description": "Abdomen appearing swollen."},
    "swelled_lymph_nodes": {"name": "Swollen Lymph Nodes", "description": "Enlarged lymph nodes in the neck, armpits, or groin."},
    "malaise": {"name": "General Discomfort", "description": "A feeling of overall unease or illness."},
    "blurred_and_distorted_vision": {"name": "Blurred Vision", "description": "Difficulty focusing or seeing clearly."},
    "phlegm": {"name": "Phlegm", "description": "Thick mucus produced by the respiratory system."},
    "throat_irritation": {"name": "Throat Irritation", "description": "Scratchy or sore throat."},
    "redness_of_eyes": {"name": "Red Eyes", "description": "Eyes appearing red or bloodshot."},
    "sinus_pressure": {"name": "Sinus Pressure", "description": "Pressure or pain in the sinuses."},
    "runny_nose": {"name": "Runny Nose", "description": "Excess nasal discharge."},
    "congestion": {"name": "Nasal Congestion", "description": "Blocked or stuffy nose."},
    "chest_pain": {"name": "Chest Pain", "description": "Pain or discomfort in the chest area."},
    "weakness_in_limbs": {"name": "Weak Limbs", "description": "Weakness in arms or legs."},
    "fast_heart_rate": {"name": "Rapid Heartbeat", "description": "Heart beating faster than normal."},
    "pain_during_bowel_movements": {"name": "Bowel Movement Pain", "description": "Pain while passing stools."},
    "pain_in_anal_region": {"name": "Anal Pain", "description": "Pain or discomfort in the anal area."},
    "bloody_stool": {"name": "Blood in Stool", "description": "Presence of blood in feces."},
    "irritation_in_anus": {"name": "Anal Irritation", "description": "Itching or discomfort around the anus."},
    "neck_pain": {"name": "Neck Pain", "description": "Ache or discomfort in the neck."},
    "dizziness": {"name": "Dizziness", "description": "Feeling lightheaded or unsteady."},
    "cramps": {"name": "Muscle Cramps", "description": "Sudden, involuntary muscle contractions."},
    "bruising": {"name": "Bruising", "description": "Discoloration of the skin due to injury."},
    "obesity": {"name": "Obesity", "description": "Excessive body weight."},
    "swollen_legs": {"name": "Swollen Legs", "description": "Legs appearing puffy or enlarged."},
    "swollen_blood_vessels": {"name": "Swollen Blood Vessels", "description": "Enlarged veins or arteries."},
    "puffy_face_and_eyes": {"name": "Puffy Face", "description": "Swelling in the face and around the eyes."},
    "enlarged_thyroid": {"name": "Enlarged Thyroid", "description": "Swollen thyroid gland in the neck."},
    "brittle_nails": {"name": "Brittle Nails", "description": "Weak or easily broken nails."},
    "swollen_extremities": {"name": "Swollen Extremities", "description": "Swelling in hands, feet, or limbs."},
    "excessive_hunger": {"name": "Excessive Hunger", "description": "Increased appetite despite eating."},
    "extra_marital_contacts": {"name": "Unprotected Intimacy", "description": "Engaging in unprotected sexual activity outside marriage."},
    "drying_and_tingling_lips": {"name": "Dry Lips", "description": "Lips feeling dry or tingly."},
    "slurred_speech": {"name": "Slurred Speech", "description": "Difficulty speaking clearly."},
    "knee_pain": {"name": "Knee Pain", "description": "Ache or discomfort in the knees."},
    "hip_joint_pain": {"name": "Hip Pain", "description": "Pain in the hip joints."},
    "muscle_weakness": {"name": "Muscle Weakness", "description": "Reduced strength in muscles."},
    "stiff_neck": {"name": "Stiff Neck", "description": "Difficulty moving the neck due to tightness."},
    "swelling_joints": {"name": "Swollen Joints", "description": "Joints appearing puffy or enlarged."},
    "movement_stiffness": {"name": "Stiff Movement", "description": "Difficulty moving joints smoothly."},
    "spinning_movements": {"name": "Spinning Sensation", "description": "Feeling like the room is spinning."},
    "loss_of_balance": {"name": "Loss of Balance", "description": "Difficulty maintaining balance."},
    "unsteadiness": {"name": "Unsteadiness", "description": "Feeling unstable or wobbly."},
    "weakness_of_one_body_side": {"name": "One-Sided Weakness", "description": "Weakness on one side of the body."},
    "loss_of_smell": {"name": "Loss of Smell", "description": "Inability to detect odors."},
    "bladder_discomfort": {"name": "Bladder Pain", "description": "Discomfort or pain in the bladder area."},
    "foul_smell_of urine": {"name": "Foul-Smelling Urine", "description": "Urine with a strong, unpleasant odor."},
    "continuous_feel_of_urine": {"name": "Frequent Urge to Urinate", "description": "Constant need to urinate."},
    "passage_of_gases": {"name": "Passing Gas", "description": "Excessive gas release from the digestive system."},
    "internal_itching": {"name": "Internal Itching", "description": "Itching sensation inside the body."},
    "toxic_look_(typhos)": {"name": "Toxic Appearance", "description": "Pale or sickly appearance."},
    "depression": {"name": "Depression", "description": "Persistent sadness or low mood."},
    "irritability": {"name": "Irritability", "description": "Easily annoyed or frustrated."},
    "muscle_pain": {"name": "Muscle Ache", "description": "Pain in the muscles."},
    "altered_sensorium": {"name": "Confusion", "description": "Altered mental state or confusion."},
    "red_spots_over_body": {"name": "Red Spots", "description": "Small red spots appearing on the skin."},
    "belly_pain": {"name": "Belly Ache", "description": "Pain in the stomach area."},
    "abnormal_menstruation": {"name": "Irregular Periods", "description": "Unusual menstrual cycles."},
    "dischromic_patches": {"name": "Skin Discoloration", "description": "Patches of discolored skin."},
    "watering_from_eyes": {"name": "Watery Eyes", "description": "Excessive tearing from the eyes."},
    "increased_appetite": {"name": "Increased Hunger", "description": "Feeling hungry more often than usual."},
    "polyuria": {"name": "Frequent Urination", "description": "Urinating more frequently than normal."},
    "family_history": {"name": "Family Medical History", "description": "Known medical conditions in the family."},
    "mucoid_sputum": {"name": "Mucus in Sputum", "description": "Thick, slimy substance in coughed-up mucus."},
    "rusty_sputum": {"name": "Rusty-Colored Sputum", "description": "Sputum with a reddish-brown color."},
    "lack_of_concentration": {"name": "Poor Concentration", "description": "Difficulty focusing or paying attention."},
    "visual_disturbances": {"name": "Vision Problems", "description": "Issues with eyesight or vision."},
    "receiving_blood_transfusion": {"name": "Blood Transfusion", "description": "Receiving donated blood."},
    "receiving_unsterile_injections": {"name": "Unsterile Injections", "description": "Injections given without proper sterilization."},
    "coma": {"name": "Coma", "description": "Deep unconsciousness."},
    "stomach_bleeding": {"name": "Stomach Bleeding", "description": "Bleeding in the stomach area."},
    "distention_of_abdomen": {"name": "Swollen Abdomen", "description": "Abdomen appearing bloated or distended."},
    "history_of_alcohol_consumption": {"name": "Alcohol Use", "description": "Regular consumption of alcohol."},
    "fluid_overload": {"name": "Fluid Retention", "description": "Swelling due to excess fluid in the body."},
    "blood_in_sputum": {"name": "Blood in Sputum", "description": "Presence of blood in coughed-up mucus."},
    "prominent_veins_on_calf": {"name": "Visible Veins", "description": "Veins prominently visible on the calf."},
    "palpitations": {"name": "Heart Palpitations", "description": "Fluttering or pounding sensation in the chest."},
    "painful_walking": {"name": "Pain While Walking", "description": "Discomfort or pain during walking."},
    "pus_filled_pimples": {"name": "Pus-Filled Pimples", "description": "Small bumps on the skin filled with pus."},
    "blackheads": {"name": "Blackheads", "description": "Small dark spots on the skin caused by clogged pores."},
    "scurring": {"name": "Scarring", "description": "Marks left on the skin after healing."},
    "skin_peeling": {"name": "Peeling Skin", "description": "Layers of skin shedding off."},
    "silver_like_dusting": {"name": "Silver Dusting", "description": "Skin with a silvery sheen."},
    "small_dents_in_nails": {"name": "Nail Dents", "description": "Tiny indentations on the nails."},
    "inflammatory_nails": {"name": "Inflamed Nails", "description": "Redness or swelling around the nails."},
    "blister": {"name": "Blisters", "description": "Fluid-filled bumps on the skin."},
    "red_sore_around_nose": {"name": "Red Sores Around Nose", "description": "Painful red sores near the nose."},
    "yellow_crust_ooze": {"name": "Yellow Crust", "description": "Crusty yellow discharge from wounds."}
}

# List of health conditions (41 outputs)
HEALTH_CONDITIONS = [
    "Fungal Infection", "Allergy", "Gerd", "Chronic Cholestasis", "Drug Reaction",
    "Peptic Ulcer Disease", "Aids", "Diabetes", "Gastroenteritis", "Bronchial Asthma",
    "Hypertension", "Migraine", "Cervical Spondylosis", "Paralysis (Brain Haemorrhage)",
    "Jaundice", "Malaria", "Chickenpox", "Dengue", "Typhoid", "Hepatitis A", "Hepatitis B",
    "Hepatitis C", "Hepatitis D", "Hepatitis E", "Alcoholic Hepatitis", "Tuberculosis",
    "Common Cold", "Pneumonia", "Dimorphic Haemorrhoids (Piles)", "Heart Attack",
    "Varicose Veins", "Hypothyroidism", "Hyperthyroidism", "Hypoglycaemia", "Osteoarthritis",
    "Arthritis", "Vertigo", "Acne", "Urinary Tract Infection", "Psoriasis", "Impetigo"
]
# Helper function to categorize symptoms
def get_category(symptom):
    if "skin" in symptom or symptom in ["itching", "rash", "nodal", "pus", "blackheads", "peeling", "silver", "nails", "inflammatory", "blister", "sore", "crust"]:
        return "skin"
    elif "stomach" in symptom or "tongue" in symptom or symptom in ["vomiting", "burning", "spotting", "indigestion", "constipation", "abdominal", "diarrhoea", "belly", "swelling"]:
        return "digestive"
    elif "sneezing" in symptom or "cough" in symptom or "breath" in symptom or "sinus" in symptom or "nose" in symptom or "chest" in symptom or "phlegm" in symptom:
        return "respiratory"
    elif "joint" in symptom or "muscle" in symptom or "knee" in symptom or "hip" in symptom or "neck" in symptom or "back" in symptom or "stiff" in symptom or "swelling" in symptom:
        return "muscular"
    elif "dizziness" in symptom or "spinning" in symptom or "balance" in symptom or "unsteady" in symptom or "weakness" in symptom or "smell" in symptom or "speech" in symptom:
        return "neurological"
    elif "fatigue" in symptom or "weight" in symptom or "restless" in symptom or "lethargy" in symptom or "malaise" in symptom or "dehydration" in symptom or "sweating" in symptom:
        return "general"
    elif "urination" in symptom or "bladder" in symptom or "urine" in symptom or "polyuria" in symptom:
        return "urinary"
    elif "vision" in symptom or "eyes" in symptom or "watering" in symptom:
        return "vision"
    elif "anxiety" in symptom or "mood" in symptom or "depression" in symptom or "irritability" in symptom or "concentration" in symptom:
        return "mental"
    else:
        return "other"
@app.route('/')
def index():
    return render_template('index.html', symptoms=SYMPTOM_DETAILS, get_category=get_category)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    selected_symptoms = data['symptoms']
    print("Received Symptoms:", selected_symptoms)  # Debugging log

    # Create a one-hot encoded input vector
    input_vector = np.zeros(len(ORIGINAL_SYMPTOMS))
    for symptom in selected_symptoms:
        if symptom in ORIGINAL_SYMPTOMS:
            input_vector[ORIGINAL_SYMPTOMS.index(symptom)] = 1

    # Predict using the TensorFlow model
    input_tensor = tf.convert_to_tensor([input_vector], dtype=tf.float32)
    predictions = model(input_tensor)
    predictions = tf.nn.softmax(predictions)
    predicted_index = tf.argmax(predictions, axis=1).numpy()[0]

    # Ensure the predicted index is within bounds
    if predicted_index < len(HEALTH_CONDITIONS):
        predicted_condition = HEALTH_CONDITIONS[predicted_index]
    else:
        predicted_condition = "Unknown Condition"

    # Return a softer, more user-friendly message
    message = (
        f"Based on the symptoms you've described, it seems like you might be experiencing "
        f"'{predicted_condition}'. However, this is not a definitive diagnosis. Please consult "
        f"a healthcare professional for accurate advice and treatment."
    )
    print("Final Message:", message)  # Debugging log
    return jsonify({'message': message})
# Configure Gemini API Key
os.environ['GOOGLE_API_KEY'] = 'YOUR_GOOGLE_GEMINI_API_KEY'
genai.configure(api_key='AIzaSyB4Lyib3DSSroqRIJqISqLbomzRxZKpIUY')

# Load the model
model_ = genai.GenerativeModel('gemini-pro')

# Define prompts for each health condition
PROMPTS = {

    0: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Fungal Infection], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    1: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Allergy], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    2: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Gerd], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    3: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Chronic Cholestasis], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    4: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Drug Reaction], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    5: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Peptic Ulcer Disease], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    6: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Aids], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    7: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Diabetes], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    8: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Gastroenteritis], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    9: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Bronchial Asthma], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    10: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Hypertension], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    11: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Migraine], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    12: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Cervical Spondylosis], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    13: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Paralysis (Brain Haemorrhage)], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    14: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Jaundice], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    15: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Malaria], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    16: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Chickenpox], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    17: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Dengue], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    18: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Typhoid], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    19: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Hepatitis A], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    20: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Hepatitis B], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    21: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Hepatitis C], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    22: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Hepatitis D], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    23: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Hepatitis E], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    24: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Alcoholic Hepatitis], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    25: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Tuberculosis], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    26: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Common Cold], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    27: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Pneumonia], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    28: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Dimorphic Haemorrhoids (Piles)], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    29: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Heart Attack], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    30: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Varicose Veins], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    31: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Hypothyroidism], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    32: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Hyperthyroidism], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    33: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Hypoglycaemia], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    34: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Osteoarthritis], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    35: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Arthritis], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    36: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Vertigo], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    37: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Acne], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    38: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Urinary Tract Infection], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    39: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Psoriasis], Symptoms :[SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
    40: "You are an AI healthcare assistant. Based on the following details, provide (1) general recommendations for managing or improving the condition, including actionable advice or lifestyle changes, and (2) a list of nearby doctors or specialists who treat the condition, including their names, specialties, and contact details if possible; if exact details aren’t available, suggest steps to locate specialists in the area. Input details: Condition/Disease : [Impetigo], Symptoms : [SYMPTOMS_LIST], Location : [LOCATION]. Ensure your response is clear, concise, and tailored to the user's needs.",
}


@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_message = data.get('message', '').strip()
    location = data.get('location', 'Unknown Location')
    symptoms = data.get('symptoms', [])

    # Map user input to a health condition index (e.g., based on prediction)
    condition_index = int(data.get('condition_index', 0))  # Default to 0 (Fungal Infection)

    # Replace placeholders in the prompt
    prompt = PROMPTS.get(condition_index, "").replace(
        "[SYMPTOMS_LIST]", ", ".join(symptoms)
    ).replace("[LOCATION]", location)

    # Generate response using Gemini API
    try:
        response = model_.generate_content(prompt)
        bot_response = response.text.strip()
    except Exception as e:
        bot_response = f"An error occurred while generating a response: {str(e)}"

    return jsonify({'response': bot_response})
if __name__ == '__main__':
    app.run(debug=True)

