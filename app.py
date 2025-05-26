import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import random # Import the random module

# --- Page Configuration ---
st.set_page_config(
    page_title="Student CGPA Predictor & Advisor",
    page_icon="🧑‍🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Models and Encoders ---
MODEL_DIR = "Models"

@st.cache_data
def load_pickle(file_name):
    """Loads a pickle file from the MODEL_DIR."""
    file_path = os.path.join(MODEL_DIR, file_name)
    if not os.path.exists(file_path):
        st.error(f"Error: File '{file_name}' not found in '{MODEL_DIR}' directory.")
        st.info("Please ensure all .pkl files (model and encoders) are in the 'Models' directory.")
        return None
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading {file_name}: {e}")
        return None

# Load the main prediction model (ExtraTreesRegressor)
model = load_pickle('extratreesRegressor.pkl')

# Load all encoders
encoders = {
    "Students age": load_pickle('encoder_Students_age.pkl'),
    "Gender": load_pickle('encoder_Gender.pkl'),
    "Type of University": load_pickle('encoder_Type_of_University.pkl'),
    "Father's education": load_pickle('encoder_Father_education.pkl'),
    "Father's Occupation": load_pickle('encoder_Father_occupation.pkl'),
    "Mother's Occupation": load_pickle('encoder_Mother_occupation.pkl'),
    "Family Income(monthly)": load_pickle('encoder_Family_Income_Monthly.pkl'),
    "Number of Siblings": load_pickle('encoder_Number_of_siblings.pkl'),
    "Parental status": load_pickle('encoder_Parental_status.pkl'),
    "SSC result": load_pickle('encoder_SSC_result.pkl'),
    "HSC result": load_pickle('encoder_HSC_result.pkl'),
    "Scholarship in SSC": load_pickle('encoder_Scholarship_in_SSC.pkl'),
    "Scholarship in HSC": load_pickle('encoder_Scholarship_in_HSC.pkl'),
    "Accommodation": load_pickle('encoder_Accommodation.pkl'),
    "Weekly Study Time at home": load_pickle('encoder_Weekly_study_time_at_home.pkl'),
    "Reading Scientific books/articles/journals": load_pickle('encoder_Reading_Scientific_books.pkl'),
    "Reading Non-Scientific books/articles/journals": load_pickle('encoder_Reading_Non_Scientific_books.pkl'),
    "Attendance in class": load_pickle('encoder_Attendance_in_class.pkl'),
    "Mid-Term Exam Preparation": load_pickle('encoder_Mid_Term_Exam_Preparation.pkl'),
    "Taking Exam Preparation": load_pickle('encoder_Taking_Exam_Preparation.pkl'),
    "Taking Class Note": load_pickle('encoder_Taking_Class_Note.pkl'),
    "Any Co-curricular activity": load_pickle('encoder_Any_Co_curricular_activity.pkl'),
    "Attend any Seminar related to department": load_pickle('encoder_Attend_any_Seminar_related_to_department.pkl'),
    "Any part-time job": load_pickle('encoder_Any_part_time_job.pkl'),
    "Undergraduate 1st semester result (CGPA/GPA out of 4)": load_pickle('encoder_Undergraduate_1st_semester_result.pkl')
}

essential_files_loaded = model is not None and all(encoders.get(k) is not None for k in encoders)


# --- Personalized Advice Function ---
def generate_personalized_advice(predicted_cgpa_category, user_inputs):
    advice_list = []

    # 1. General Advice based on Predicted CGPA
    cgpa_advice_pools = {
        "Fail": [
            "This result can be a turning point. Identify areas of difficulty and seek help from professors or academic advisors. Don't be discouraged; many successful people have faced setbacks.",
            "It's important to understand what went wrong. Reflect on your study habits, attendance, and understanding of the material. Create a solid plan for the next semester.",
            "Remember, one result doesn't define your potential. Use this as a learning experience. Focus on building a stronger foundation and developing effective study strategies."
        ],
        "Less than 2.50": [
            "This is an opportunity to significantly improve. Focus on understanding core concepts and practice regularly. Consider forming study groups or seeking tutoring.",
            "Your current CGPA indicates a need for a strategic shift. Prioritize your subjects, manage your time effectively, and don't hesitate to ask for clarification in class.",
            "Building a stronger academic foundation now will be very beneficial. Review your learning methods and explore techniques like active recall and spaced repetition."
        ],
        "2.50 - 2.74": [
            "You're building a foundation. To push higher, try to deepen your understanding of complex topics and enhance your exam preparation techniques. Small consistent efforts can lead to big gains.",
            "Consider dedicating a bit more time to challenging subjects. Effective time management and consistent revision can help you move into a higher CGPA bracket.",
            "Aim for consistent improvement. Review your performance in mid-terms and assignments to identify areas where you can score better."
        ],
        "2.75 - 2.99": [
            "You're on a steady path. To improve further, focus on mastering key concepts and practicing past papers. Engaging more actively in class discussions can also be beneficial.",
            "This is a good base. Identify subjects where you can excel further. Refining your note-taking and revision strategies could provide an edge.",
            "Keep up the effort! A little more focus on weaker areas or more consistent revision could help you cross the 3.00 mark."
        ],
        "3.00 - 3.24": [
            "Well done on achieving a good CGPA! To maintain and improve, continue with consistent study habits and explore topics beyond the syllabus to deepen your interest.",
            "This is a solid performance. Consider taking on challenging projects or participating in academic competitions to further hone your skills.",
            "You're doing well. Ensure you maintain a good balance between academics and other activities to avoid burnout and stay motivated."
        ],
        "3.25 - 3.49": [
            "Excellent work! You're performing well. To aim even higher, perhaps focus on advanced problem-solving or research-oriented activities in your field.",
            "This CGPA reflects strong effort. Continue to engage deeply with your coursework and seek opportunities that stretch your understanding.",
            "Keep up the great momentum! Consider mentoring junior students or leading study groups to solidify your own understanding."
        ],
        "3.50 - 3.74": [
            "Outstanding performance! You are clearly dedicated. Explore advanced coursework or research opportunities to challenge yourself further.",
            "This is a remarkable achievement. Maintain your effective study strategies and consider sharing your learning techniques with peers.",
            "Your hard work is paying off. Continue to strive for excellence and look for ways to apply your knowledge in practical settings."
        ],
        "3.75 - 4.00": [
            "Exceptional! This is a top-tier performance. Continue to pursue your academic interests with passion and consider long-term goals like graduate studies or specialized careers.",
            "Truly impressive! Your dedication is clear. Seek out opportunities for leadership in academic projects or contribute to research in your department.",
            "You're at the pinnacle of academic performance. Mentor others, engage in cutting-edge topics, and continue to challenge yourself intellectually."
        ]
    }
    advice_list.append(random.choice(cgpa_advice_pools.get(predicted_cgpa_category, ["Focus on consistent effort and understanding your course material."])))

    # 2. Specific Feature-based Advice
    study_time = user_inputs.get("Weekly Study Time at home")
    if study_time == "Less than 1 Hour":
        advice_list.append(random.choice([
            "Dedicating even a few more hours per week to focused study could significantly improve your grasp of the subjects. Try scheduling short, regular study blocks.",
            "Consider if 'Less than 1 Hour' of weekly study is sufficient for your courses. Consistent, even if short, study sessions are often more effective than cramming."
        ]))
    elif study_time in ["2-5 Hour", "6-10Hour"]:
         advice_list.append(random.choice([
            "Good job on your current study schedule! Ensure your study time is effective by minimizing distractions and using active learning techniques.",
            "Consistency is key! If you find yourself struggling with certain topics, see if slightly increasing focused time on those areas helps."
        ]))
    elif study_time in ["11-14 Hour", "11-15 Hour", "More than 15 Hour"]: # Note: '11-15 Hour' was in your list, I combined it
        advice_list.append(random.choice([
            "That's a significant amount of study time! Make sure you're also scheduling breaks to prevent burnout and maintain peak concentration.",
            "With dedicated study hours, focus on 'studying smart, not just hard'. Are your methods efficient? Are you practicing enough problems?"
        ]))

    if user_inputs.get("Taking Class Note") == "Never":
        advice_list.append(random.choice([
            "Taking notes in class, even brief ones, can greatly improve retention and understanding. It helps you stay engaged and provides material for revision.",
            "If you're not taking notes, you might be missing out on key points from lectures. Try a simple note-taking method to see if it helps."
        ]))
    elif user_inputs.get("Taking Class Note") == "Sometimes":
        advice_list.append(random.choice([
            "Making your class note-taking more consistent could provide a stronger foundation for exam preparation. Identify what makes you take notes sometimes and try to expand on that.",
            "Good that you take notes sometimes! Try to be more regular, especially for complex topics or when the instructor highlights important concepts."
        ]))


    if user_inputs.get("Attendance in class") == "Less than 60% classes":
        advice_list.append(random.choice([
            "Attending classes regularly is crucial. You miss out on explanations, discussions, and important announcements. Aim to improve your attendance.",
            "Low class attendance can directly impact your understanding and grades. Try to make attending lectures a priority."
        ]))

    if user_inputs.get("Taking Exam Preparation") == "Close date to exam":
        advice_list.append(random.choice([
            "Relying on last-minute preparation can be stressful and less effective. Try to study regularly throughout the semester for better understanding and retention.",
            "Consistent study throughout the semester, rather than just before exams, often leads to better long-term learning and less stress."
        ]))

    if user_inputs.get("Any Co-curricular activity") == "No":
        advice_list.append(random.choice([
            "Engaging in co-curricular activities can be a great way to develop new skills, network, and de-stress. Explore clubs or activities that interest you.",
            "University life is also about holistic development. Consider joining a co-curricular activity for a well-rounded experience and to enhance your soft skills."
        ]))

    if user_inputs.get("Reading Scientific books/articles/journals") == "No" and user_inputs.get("Reading Non-Scientific books/articles/journals") == "No":
        advice_list.append(random.choice([
            "Expanding your reading beyond textbooks can broaden your perspective and critical thinking skills. Try picking up a book or article on a topic you're curious about.",
            "Reading, whether scientific or non-scientific, enriches your knowledge and vocabulary. It could be a relaxing and beneficial habit to cultivate."
        ]))
    elif user_inputs.get("Reading Scientific books/articles/journals") == "No":
        advice_list.append(random.choice([
            "While you read non-scientific material, incorporating some scientific reading related to your field or interests can deepen your subject knowledge and critical analysis.",
            "Exploring scientific articles or books, even popular science, can be very beneficial for understanding advancements and methodologies in various fields."
        ]))


    if user_inputs.get("Any part-time job") not in ["Not Applicable", None]: # Assuming 'None' if not selected
        advice_list.append(random.choice([
            "Balancing a part-time job with studies requires excellent time management. Ensure you're allocating enough time for both without compromising your well-being.",
            "Having a part-time job is commendable. Make sure to schedule dedicated study periods and rest to avoid burnout and maintain academic performance."
        ]))

    # 3. Time Management Advice (General)
    time_management_tips = [
        "Effective time management is key to academic success. Try using a planner or a digital calendar to schedule study sessions, assignments, and breaks.",
        "Break down large academic tasks into smaller, more manageable steps. This can make them feel less daunting and easier to accomplish.",
        "Prioritize your tasks. Focus on the most important and urgent ones first. Techniques like the Eisenhower Matrix (Urgent/Important) can be helpful.",
        "Minimize distractions during your study time. Find a quiet place and consider using tools or apps to block distracting websites or notifications.",
        "Don't forget to schedule regular breaks during long study sessions. Short breaks can help you stay focused and improve overall productivity."
    ]
    advice_list.append(random.choice(time_management_tips))
    if len(advice_list) < 5 : # Ensure at least a few tips
        advice_list.append(random.choice(time_management_tips))


    return list(set(advice_list)) # Return unique advice, randomly pick around 3-5
    # For more controlled variety, you might sample a fixed number
    # return random.sample(list(set(advice_list)), k=min(len(list(set(advice_list))), 5))



# --- UI Elements ---
st.title("🧑‍🎓 Student Performance Predictor & Advisor")
st.markdown("Predict a student's 1st Semester CGPA and get personalized feedback.")
st.markdown("---")

if essential_files_loaded:
    st.sidebar.header("Student Information")
    input_data_for_model = {} # For encoded values
    input_data_for_advice = {} # For original string values

    # Corrected feature_order list based on typical model training output
    feature_order = [
        'Students age', 'Gender', 'Type of University', "Father's education",
        "Father's Occupation", "Mother's Occupation",
        "Family Income(monthly)", "Number of Siblings", "Parental status",
        "SSC result", "HSC result", "Scholarship in SSC", "Scholarship in HSC",
        "Accommodation", "Weekly Study Time at home",
        "Reading Scientific books/articles/journals",
        "Reading Non-Scientific books/articles/journals", "Attendance in class",
        "Mid-Term Exam Preparation", "Taking Exam Preparation",  "Taking Class Note",
        "Any Co-curricular activity", "Attend any Seminar related to department",
        "Any part-time job"
    ]


    def create_selectbox(feature_name, default_index=0):
        encoder = encoders.get(feature_name)
        if encoder:
            options = list(encoder.categories_[0])
            selected_value = st.sidebar.selectbox(
                label=f"Select {feature_name}:",
                options=options,
                index=default_index,
                key=f"select_{feature_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '').lower()}"
            )
            input_data_for_advice[feature_name] = selected_value # Store string for advice
            # For the model, we will encode this later if needed, or store directly if already used for encoding
            # For now, let's assume we re-encode just before prediction
        else:
            st.sidebar.warning(f"Encoder for '{feature_name}' not found. This feature won't be used.")

    with st.sidebar.expander("👤 Personal & Family Background", expanded=True):
        create_selectbox("Students age", default_index=0)
        create_selectbox("Gender", default_index=0)
        create_selectbox("Type of University", default_index=0)
        create_selectbox("Father's education", default_index=2)
        create_selectbox("Mother's Occupation", default_index=4)
        create_selectbox("Father's Occupation", default_index=1)
        create_selectbox("Family Income(monthly)", default_index=1)
        create_selectbox("Number of Siblings", default_index=1)
        create_selectbox("Parental status", default_index=0)
        create_selectbox("Accommodation", default_index=0)

    with st.sidebar.expander("📚 Academic Background", expanded=True):
        create_selectbox("SSC result", default_index=2)
        create_selectbox("HSC result", default_index=2)
        create_selectbox("Scholarship in SSC", default_index=0)
        create_selectbox("Scholarship in HSC", default_index=0)

    with st.sidebar.expander("📖 Study Habits & Engagement", expanded=True):
        create_selectbox("Taking Class Note", default_index=1)
        create_selectbox("Weekly Study Time at home", default_index=1)
        create_selectbox("Reading Scientific books/articles/journals", default_index=1)
        create_selectbox("Reading Non-Scientific books/articles/journals", default_index=1)
        create_selectbox("Attendance in class", default_index=1)
        create_selectbox("Mid-Term Exam Preparation", default_index=1)
        create_selectbox("Taking Exam Preparation", default_index=1)
        create_selectbox("Any Co-curricular activity", default_index=0)
        create_selectbox("Attend any Seminar related to department", default_index=0)
        create_selectbox("Any part-time job", default_index=0)

    if st.sidebar.button("✨ Predict CGPA & Get Advice", use_container_width=True, type="primary"):
        encoded_inputs_for_model = []
        all_inputs_valid = True
        
        for feature in feature_order:
            user_string_value = input_data_for_advice.get(feature) # Get the string value
            encoder = encoders.get(feature)

            if user_string_value is not None and encoder is not None:
                try:
                    # Create DataFrame for transform as it expects feature names
                    input_df_for_transform = pd.DataFrame([[user_string_value]], columns=[feature])
                    encoded_value = encoder.transform(input_df_for_transform)[0, 0]
                    encoded_inputs_for_model.append(int(encoded_value))
                except Exception as e:
                    st.error(f"Error encoding '{feature}' with value '{user_string_value}': {e}")
                    all_inputs_valid = False
                    break
            elif feature in feature_order: # Only error if it's a required feature for the model
                st.error(f"Missing input for required model feature: {feature}")
                all_inputs_valid = False
                break
        
        if all_inputs_valid and len(encoded_inputs_for_model) == len(feature_order):
            prediction_input_df = pd.DataFrame([encoded_inputs_for_model], columns=feature_order)
            
            # st.subheader("Processed Input Data (Encoded Values Sent to Model):")
            # st.dataframe(prediction_input_df.style.format("{:.0f}")) # Optional: for debugging

            try:
                prediction_numeric_raw = model.predict(prediction_input_df)
                predicted_cgpa_encoded_from_model = int(round(prediction_numeric_raw[0]))

                target_encoder = encoders["Undergraduate 1st semester result (CGPA/GPA out of 4)"]
                target_feature_name = "Undergraduate 1st semester result (CGPA/GPA out of 4)"
                
                min_target_code = 0
                max_target_code = len(target_encoder.categories_[0]) - 1
                
                final_predicted_code_for_inverse_transform = predicted_cgpa_encoded_from_model
                warning_message = None

                if predicted_cgpa_encoded_from_model < min_target_code:
                    final_predicted_code_for_inverse_transform = min_target_code
                    # warning_message kept internal, advice will reflect it
                elif predicted_cgpa_encoded_from_model > max_target_code:
                    final_predicted_code_for_inverse_transform = max_target_code

                input_df_for_inverse_transform = pd.DataFrame(
                    [[final_predicted_code_for_inverse_transform]],
                    columns=[target_feature_name]
                )
                predicted_cgpa_category_string = target_encoder.inverse_transform(input_df_for_inverse_transform)[0,0]

                st.markdown("---")
                st.subheader("📈 Predicted 1st Semester CGPA Range:")
                st.success(f"**{predicted_cgpa_category_string}**")
                
                # st.info(f"Raw model prediction (continuous): {prediction_numeric_raw[0]:.2f}") # Optional

                # --- Generate and Display Advice ---
                st.markdown("---")
                st.subheader("💡 Personalized Feedback & Suggestions:")
                
                # Pass the original string inputs for advice generation
                generated_advice = generate_personalized_advice(predicted_cgpa_category_string, input_data_for_advice)
                
                if generated_advice:
                    # To ensure variety, shuffle and pick a few if many are generated
                    # random.shuffle(generated_advice)
                    # displayed_advice = generated_advice[:min(len(generated_advice), 5)] # Display up to 5 tips
                    
                    # For now, let's display all unique ones generated
                    for i, adv in enumerate(generated_advice):
                        st.markdown(f"""
                        <div style="margin-bottom: 10px; padding: 10px; border-left: 5px solid #007bff; background-color: #f8f9fa; border-radius: 5px; color: black;">
                            <strong>Suggestion {i+1}:</strong> {adv}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No specific additional advice generated for this profile yet.")

            except ValueError as ve:
                st.error(f"Model Prediction Error: {ve}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        elif all_inputs_valid and len(encoded_inputs_for_model) != len(feature_order):
             st.error(f"Input Mismatch: Encoded {len(encoded_inputs_for_model)} inputs, but model expects {len(feature_order)}.")
        # else: # Covered by all_inputs_valid check
            # st.error("Could not make a prediction due to issues with input data or encoders.")

    st.sidebar.markdown("---")
    st.sidebar.info("This tool provides predictions and suggestions based on a statistical model. Individual results may vary.")

else:
    st.error("🚨 Application Critical Error: Essential model or encoder files are missing.")
    # ... (rest of the error handling for missing files)

st.markdown("---")
st.markdown("<div style='text-align: center; color: grey; font-size: small;'>App by AI Student Advisor Team | SKS</div>", unsafe_allow_html=True)

