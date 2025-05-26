import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Student CGPA Predictor",
    page_icon="🎓",
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


# --- UI Elements ---
st.title("🎓 Student Performance Predictor")
st.markdown("Predict a student's 1st Semester Undergraduate CGPA based on various factors.")
st.markdown("---")

if essential_files_loaded:
    st.sidebar.header("Student Information")
    input_data = {}

    # -----------------------------------------------------------------------------
    # CRITICAL: This list MUST match the feature names and order from x_train.columns
    #           in your model training script.
    # Get this by printing `list(x_train.columns)` in your training script
    # right before `model.fit(x_train, y_train)`.
    # -----------------------------------------------------------------------------
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
    # -----------------------------------------------------------------------------

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
            input_data[feature_name] = selected_value
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

    if st.sidebar.button("✨ Predict CGPA", use_container_width=True, type="primary"):
        encoded_inputs = []
        all_inputs_valid = True
        
        for feature in feature_order:
            user_value = input_data.get(feature)
            encoder = encoders.get(feature)
            if user_value is not None and encoder is not None:
                try:
                    input_df_for_transform = pd.DataFrame([[user_value]], columns=[feature])
                    encoded_value = encoder.transform(input_df_for_transform)[0, 0]
                    encoded_inputs.append(int(encoded_value))
                except Exception as e:
                    st.error(f"Error encoding '{feature}' with value '{user_value}': {e}")
                    all_inputs_valid = False
                    break
            elif feature in feature_order: # Only error if it's a required feature
                st.error(f"Missing input or encoder for required feature: {feature}")
                all_inputs_valid = False
                break
        
        if all_inputs_valid and len(encoded_inputs) == len(feature_order):
            # This DataFrame MUST have columns in the same order and with the same names
            # as the DataFrame used to fit the 'model'.
            prediction_input_df = pd.DataFrame([encoded_inputs], columns=feature_order)
            
            st.subheader("Processed Input Data (Encoded Values Sent to Model):")
            st.dataframe(prediction_input_df.style.format("{:.0f}"))

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
                    warning_message = (f"Model prediction ({prediction_numeric_raw[0]:.2f} -> {predicted_cgpa_encoded_from_model}) "
                                       f"was below the lowest CGPA category. Capped to category code {final_predicted_code_for_inverse_transform}.")
                elif predicted_cgpa_encoded_from_model > max_target_code:
                    final_predicted_code_for_inverse_transform = max_target_code
                    warning_message = (f"Model prediction ({prediction_numeric_raw[0]:.2f} -> {predicted_cgpa_encoded_from_model}) "
                                       f"was above the highest CGPA category. Capped to category code {final_predicted_code_for_inverse_transform}.")

                if warning_message:
                    st.warning(warning_message)

                input_df_for_inverse_transform = pd.DataFrame(
                    [[final_predicted_code_for_inverse_transform]],
                    columns=[target_feature_name]
                )
                predicted_cgpa_category_string = target_encoder.inverse_transform(input_df_for_inverse_transform)[0,0]

                st.markdown("---")
                st.subheader("Predicted 1st Semester CGPA Range:")
                st.success(f"**{predicted_cgpa_category_string}**")
                
                st.info(f"Raw model prediction (continuous): {prediction_numeric_raw[0]:.2f}\n"
                        f"Rounded to category code: {predicted_cgpa_encoded_from_model}\n"
                        f"Final code used for category lookup: {final_predicted_code_for_inverse_transform}")

            except ValueError as ve: # Catch the specific error
                st.error(f"Model Prediction Error: {ve}")
                st.error("This usually means the feature names or their order in the input data "
                         "do not match what the model was trained on. "
                         "Please verify the 'feature_order' list in the app code "
                         "against 'x_train.columns' from your training script.")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction or inverse transformation: {e}")
        elif all_inputs_valid and len(encoded_inputs) != len(feature_order):
             st.error(f"Mismatch in number of encoded inputs ({len(encoded_inputs)}) and expected features ({len(feature_order)}). Please check feature processing.")
        else:
            st.error("Could not make a prediction due to issues with input data or encoders.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed for Student Performance Analysis.")

else:
    st.error(
        "🚨 Application Critical Error: Essential model or encoder files are missing. "
        "Please ensure all `.pkl` files are correctly placed in the 'Models' directory "
        "and that the filenames in the script match the actual files."
    )
    st.info(f"Expected model file: `final_extratrees_model.pkl` in `{MODEL_DIR}`")
    missing_encoders_msg = "Expected encoder files (e.g., "
    missing_encoders = [name for name, enc in encoders.items() if enc is None]
    if missing_encoders:
        missing_encoders_msg += ", ".join([f"`encoder_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '')}.pkl`" for name in missing_encoders])
        st.info(missing_encoders_msg + f") in `{MODEL_DIR}` seem to be missing or failed to load.")
    else:
        st.info(f"All declared encoders seem to be loaded, but the 'model' might be missing or another issue occurred.")


st.markdown("---")
st.markdown("<div style='text-align: center; color: grey; font-size: small;'>App by Your Name/Team | Powered by Streamlit</div>", unsafe_allow_html=True)

