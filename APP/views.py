import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # ✅ Disable OneDNN optimizations
import tensorflow as tf # type: ignore
import pickle
import numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.hashers import make_password
from .models import Patient, Lab, TestDetails, Feedback, TestRequest, Bill
from django.contrib.auth.hashers import check_password
from django.http import HttpResponse
from .models import Prediction  # ✅ Import the Prediction model
from django.utils.timezone import now  # ✅ Import now for timestamps
import tensorflow as tf
from django.conf import settings
import razorpay
from django.conf import settings
from django.shortcuts import render, get_object_or_404

@login_required
def approve_lab(request, lab_id):
    lab = get_object_or_404(Lab, id=lab_id)
    lab.is_approved = True
    lab.save()
    messages.success(request, f"Lab {lab.name} has been approved!")
    return redirect("/admin/APP/lab/")  # Redirect back to Admin panel

# ❌ Reject Lab
@login_required
def reject_lab(request, lab_id):
    lab = get_object_or_404(Lab, id=lab_id)
    lab.is_approved = False
    lab.save()
    messages.warning(request, f"Lab {lab.name} has been rejected!")
    return redirect("/admin/APP/lab/")  # Redirect back to Admin panel

# ✅ Approve Patient
@login_required
def approve_patient(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id)
    patient.is_approved = True
    patient.save()
    messages.success(request, f"Patient {patient.name} has been approved!")
    return redirect("/admin/APP/patient/")  # Redirect back to Admin panel

# ❌ Reject Patient
@login_required
def reject_patient(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id)
    patient.is_approved = False
    patient.save()
    messages.warning(request, f"Patient {patient.name} has been rejected!")
    return redirect("/admin/APP/patient/")  # Redirect back to Admin panel



# Load ML Model
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# model_path = os.path.join(BASE_DIR, "APP", "finalized_model.pkl")
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model file not found at {model_path}")
# with open(model_path, "rb") as file:
#     model = pickle.load(file)
# print("Model loaded successfully!")

ml_model = pickle.load(open(r"C:\Users\sahad\Desktop\HEART_DISEASE_PREDICTION\Heart_disease_prediction_system-current\heart_disease_prediction\finalized_model.pkl", "rb"))
#dl_model = tf.keras.models.load_model(r"C:\Users\sahad\Desktop\HEART_DISEASE_PREDICTION\Heart_disease_prediction_system-current\heart_disease_prediction\ecg_model.keras")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ecg_model.keras")

try:
    dl_model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ DL Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")

# try:
#     dl_model_path = os.path.join(settings.BASE_DIR, "heart_disease_prediction", "ecg_model.keras")
#     dl_model = tf.keras.models.load_model(dl_model_path)

#     print("✅ DL Model Loaded Successfully")
# except Exception as e:
#     print(f"❌ Error Loading Model: {e}")

def predict_disease(request, test_id):
    test = get_object_or_404(TestDetails, id=test_id)

    # ✅ Prepare ML Model Input
    ml_inputs = np.array([[
        test.patient.age,
        0 if test.patient.gender == "Male" else 1,  # Convert gender to numerical
        test.impulse,
        test.pressure_high,
        test.pressure_low,
        test.glucose,
        test.kcm,
        test.troponin
    ]])
    
    ml_prediction = ml_model.predict(ml_inputs)[0]  # ✅ Get ML prediction (0 or 1)

    # ✅ Load and Preprocess ECG Image for DL Model
    # Load and preprocess ECG image
    image_path = test.ecg_image.path  
    if not os.path.exists(image_path):
        print(f"❌ Image not found at: {image_path}")  # Debugging
    else:
        print(f"✅ Image found at: {image_path}")  # Debugging

     # Load image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Reshape for model input

    # Predict using DL Model
    dl_prediction = dl_model.predict(img_array)  
    dl_pred_label = np.argmax(dl_prediction)  # Converts softmax output to class
    dl_result = "Abnormal" if dl_pred_label == 1 else "Normal"

  

    # ✅ Save Predictions in the Database
    prediction = Prediction.objects.create(
    test=test,
    patient=test.patient,
    ml_prediction=ml_prediction,
    dl_prediction=dl_result  # ✅ Ensure DL prediction is saved
    )

    print(f"✅ Prediction Saved: ML - {prediction.ml_prediction}, DL - {prediction.dl_prediction}")  # Debugging



    return render(request, "result.html", {"prediction": prediction})

# ML Prediction
def ml_input_form(request):
    patients = Patient.objects.filter(is_approved=True)  # Get all approved patients
    if request.method == "POST":
        patient_id = request.POST.get("patient_id")

        # ✅ Fetch patient test details
        test_details = TestDetails.objects.filter(patient_id=patient_id).first()
        if not test_details:
            messages.error(request, "No test details found for this patient.")
            return redirect("ml_input_form")

        # ✅ Prepare data for ML prediction
        input_data = np.array([[  
            float(test_details.patient.age),  # ✅ Fetch from Patient model  
            0 if test_details.patient.gender == "Male" else 1,  # ✅ Fetch from Patient model  
            float(test_details.impulse),  
            float(test_details.pressure_high),  
            float(test_details.pressure_low),  
            float(test_details.glucose),  
            float(test_details.kcm),  
            float(test_details.troponin)  
        ]])

        image_path = test_details.ecg_image.path  
        if not os.path.exists(image_path):
            print(f"❌ Image not found at: {image_path}")  # Debugging
        else:
            print(f"✅ Image found at: {image_path}")  # Debugging
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ✅ Predict using the ML Model
        ml_prediction = ml_model.predict(input_data)
        ml_result = "High Risk of Heart Disease" if ml_prediction[0] == 1 else "Low Risk of Heart Disease"
        dl_prediction = dl_model.predict(img_array)  
        dl_pred_label = np.argmax(dl_prediction)  # Converts softmax output to class
        dl_result = "Abnormal" if dl_pred_label == 1 else "Normal"
        # ✅ Store Prediction Result in the Database
        prediction_entry = Prediction.objects.create(
            test=test_details,  # ✅ Link the test details
            patient=test_details.patient,  # ✅ Link the patient
            ml_prediction=ml_result,  # ✅ Store ML prediction
            dl_prediction=dl_result  # Placeholder for DL prediction
        )

        # ✅ Redirect to result page with prediction result
        return render(request, "ml_result.html", {
            "ml_result": ml_result, 
            "patient": test_details.patient,
            "prediction": prediction_entry
        })

    return render(request, "ml_input_form.html", {"patients": patients})

# Index View
def index(request):
    return render(request, "index.html")

def lab_registration(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        phone = request.POST.get("phone")
        address = request.POST.get("address")
        password = request.POST.get("password")

        if User.objects.filter(username=email).exists():
            messages.error(request, "Email is already registered.")
            return redirect("lab_registration")

        user = User.objects.create_user(username=email, email=email, password=password, first_name=name)

        lab = Lab.objects.create(user=user, phone=phone, address=address, is_approved=False)

        messages.success(request, "Lab registration successful! Wait for admin approval.")
        return redirect("lab_login")

    return render(request, "lab_registration.html")


# -----------------------------------
# 🔑 LAB LOGIN
# -----------------------------------
def lab_login(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")

        user = authenticate(request, username=email, password=password)
        if user is not None:
            try:
                lab = Lab.objects.get(user=user)
                if not lab.is_approved:
                    messages.error(request, "Your account is pending approval.")
                    return redirect("lab_login")

                login(request, user)
                request.session["lab_id"] = lab.id  # Store lab ID in session
                messages.success(request, "Login successful!")
                return redirect("lab_dash")
            except Lab.DoesNotExist:
                messages.error(request, "No lab account found.")
        else:
            messages.error(request, "Invalid email or password.")
    
    return render(request, "lab_login.html")

# -----------------------------------
# 🏥 PATIENT REGISTRATION
# -----------------------------------
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib import messages
from APP.models import Patient

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib import messages
from APP.models import Patient  # Import Patient model
from django.db import transaction

def patient_registration(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        phone = request.POST.get("phone")
        age = request.POST.get("age")
        gender = request.POST.get("gender")
        password = request.POST.get("password")

        # ✅ Check if email already exists in User table
        if User.objects.filter(username=email).exists():
            messages.error(request, "Email is already registered. Please log in.")
            return redirect("patient_registration")

        # ✅ Check if email already exists in Patient table
        if Patient.objects.filter(email=email).exists():
            messages.error(request, "A patient with this email already exists.")
            return redirect("patient_registration")

        try:
            # ✅ Ensure both User and Patient are created in one transaction
            with transaction.atomic():
                # ✅ Create User
                user = User.objects.create_user(username=email, email=email, password=password, first_name=name)
                print(f"User Created: {user}")  # Debugging Print

                # ✅ Create Patient (Linked to User)
                patient = Patient.objects.create(user=user, name=name, email=email, phone=phone, age=age, gender=gender)
                print(f"Patient Created: {patient}")  # Debugging Print

                messages.success(request, "Registration successful! Please wait for admin approval.")
                return redirect("patient_login")

        except Exception as e:
            messages.error(request, f"An error occurred: {str(e)}")
            return redirect("patient_registration")

    return render(request, "patient_registration.html")



# -----------------------------------
# 🔑 PATIENT LOGIN
# -----------------------------------
def patient_login(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")

        user = authenticate(request, username=email, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, "Login successful!")
            return redirect("patient_dash")  # Redirect to patient dashboard
        else:
            messages.error(request, "Invalid email or password.")
            return redirect("patient_login")

    return render(request, "patient_login.html")


# -----------------------------------
# 🚪 LOGOUT (For all users)
# -----------------------------------
@login_required
def user_logout(request):
    logout(request)
    messages.success(request, "Logged out successfully!")
    return redirect("index")

# Upload Test Results
def upload_test_results(request):
    if request.method == "POST":
        patient_id = request.POST.get("patient_id")
        lab_id = request.user.lab.id  # ✅ Get lab from logged-in user

        # ✅ Ensure patient & lab exist
        patient = get_object_or_404(Patient, id=patient_id)
        lab = get_object_or_404(Lab, id=lab_id)

        # ✅ Prevent duplicate test results for same patient & lab
        if TestDetails.objects.filter(patient=patient, lab=lab).exists():
            messages.error(request, "Test results for this patient have already been uploaded by your lab.")
            return redirect("upload_test_results")

        # ✅ Handle ECG Image
        ecg_image = request.FILES.get("ecg_image")

        # ✅ Save Test Details
        test = TestDetails.objects.create(
            patient=patient,
            lab=lab,  # ✅ Ensure lab is stored
            impulse=request.POST.get("impulse"),
            pressure_high=request.POST.get("pressure_high"),
            pressure_low=request.POST.get("pressure_low"),
            glucose=request.POST.get("glucose"),
            kcm=request.POST.get("kcm"),
            troponin=request.POST.get("troponin"),
            ecg_image=ecg_image
        )

        print(f"✅ Test Saved: {test} (Lab: {test.lab})")  # Debugging Print
        messages.success(request, "Test results uploaded successfully!")
        return redirect("lab_dash")

    # ✅ Fetch approved patients for selection
    patients = Patient.objects.filter(is_approved=True)
    return render(request, "upload_test_results.html", {"patients": patients})

# from django.shortcuts import render, get_object_or_404
# from django.contrib.auth.decorators import login_required
# from django.contrib import messages
# from APP.models import Lab, Patient, TestDetails

# @login_required
# def upload_test_results(request):
#     # 🔍 Debugging: Print logged-in user details
#     print("Logged-in User:", request.user)
#     print("User Email:", request.user.email)

#     # Try fetching the lab using the email of the logged-in user
#     lab = Lab.objects.filter(email=request.user.email).first()
    
#     # if not lab:
#     #     messages.error(request, "No lab account found for this user.")
#     #     return redirect("lab_login")  # Redirect back to login

#     # print("Lab Found:", lab)

#     if request.method == "POST":
#         patient_id = request.POST.get("patient_id")
#         age = request.POST.get("age")
#         gender = request.POST.get("gender")
#         impulse = request.POST.get("impulse")
#         pressure_high = request.POST.get("pressure_high")
#         pressure_low = request.POST.get("pressure_low")
#         glucose = request.POST.get("glucose")
#         kcm = request.POST.get("kcm")
#         troponin = request.POST.get("troponin")
#         ecg_image = request.FILES.get("ecg_image")

#         patient = get_object_or_404(Patient, patient_id=patient_id)

#         TestDetails.objects.create(
#             patient=patient, lab=lab,
#             age=age, gender=gender,
#             impulse=impulse, pressure_high=pressure_high,
#             pressure_low=pressure_low, glucose=glucose,
#             kcm=kcm, troponin=troponin, ecg_image=ecg_image
#         )

#         messages.success(request, "Test results uploaded successfully!")
#         return redirect("lab_dash")

#     patients = Patient.objects.filter(is_approved=True)
#     return render(request, "upload_test_results.html", {"patients": patients})

# View Test Results
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import TestDetails

@login_required
# def view_test_results(request,request_id):
#     """Fetch test results for the logged-in patient"""
#     if not hasattr(request.user, "patient"):  # Ensure the user is a patient
#         return render(request, "error.html", {"message": "You are not authorized to view this page."})

#     patient = request.user.patient
#     test_results = TestDetails.objects.filter(patient=patient).select_related("lab")  # ✅ Fetch lab info
    
#     return render(request, "view_test_results.html", {"test_results": test_results})
def view_test_results(request, request_id):
    """Fetch test results for the logged-in patient based on a particular test request."""
    # Ensure the user is a patient
    if not hasattr(request.user, "patient"):
        return render(request, "error.html", {"message": "You are not authorized to view this page."})
    
    patient = request.user.patient
    # Get the specific test request for this patient using the request_id
    test_request = get_object_or_404(TestRequest, id=request_id, patient=patient)
    # Retrieve the requested tests, assuming they're stored as a comma-separated string
    requested_tests_raw = test_request.requested_tests
    requested_tests = [test.strip() for test in requested_tests_raw.split(',')]
    
    # Fetch all test results for the patient. 
    # (You may refine this query if you want to show results only from the lab in the test_request.)
    test_results = TestDetails.objects.filter(patient=patient).select_related("lab")
    
    return render(request, "view_test_results.html", {
        "test_results": test_results,
        "requested_tests": requested_tests,
    })



# Admin Approval
def approved_patients(request):
    patients = Patient.objects.filter(is_approved=False)
    if request.method == "POST":
        patient = get_object_or_404(Patient, id=request.POST["patient_id"])
        patient.is_approved = True
        patient.save()
        return redirect("approved_patients")
    return render(request, "approved_patients.html", {"patients": patients})

def admin_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user is not None and user.is_superuser:  # Ensure only superusers can log in
            login(request, user)
            return redirect("admin_dashboard")  # Redirect to the admin dashboard
        else:
            messages.error(request, "Invalid credentials")
            return redirect("admin_login")

    return render(request, "admin_login.html")

# @login_required
# def request_test_results(request):
#     if request.method == "POST":
#         try:
#             print("Request user:", request.user)  
#             print("Request user email:", request.user.email)  

#             # ✅ Check if the patient exists
#             patient = get_object_or_404(Patient, email__iexact=request.user.email)
#             print("Patient found:", patient.name)  

#             # ✅ Get lab_id from the form
#             lab_id = request.POST.get("lab_id")
#             print("Received lab_id:", lab_id)  

#             if not lab_id:
#                 messages.error(request, "Lab ID is missing.")
#                 return redirect("request_test_results")

#             # ✅ Retrieve the lab
#             lab = get_object_or_404(Lab, lab_id=lab_id)
#             print("Lab found:", lab.name)  

#             # ✅ Create a test request
#             test_request = TestRequest.objects.create(patient=patient, lab=lab)
#             print("TestRequest created:", test_request)  

#             messages.success(request, "Test request sent successfully!")
#             return redirect("patient_dash")

#         except Patient.DoesNotExist:
#             messages.error(request, "No patient record found for this email.")
#             return redirect("request_test_results")

#         except Lab.DoesNotExist:
#             messages.error(request, "Lab record not found.")
#             return redirect("request_test_results")

#     # ✅ Get approved labs
#     available_labs = Lab.objects.filter(is_approved=True)
#     return render(request, "request_test_results.html", {"labs": available_labs})
def view_prediction_results(request):
    """Show prediction results for the logged-in patient."""
    patient = request.user.patient  # Get the logged-in patient
    predictions = Prediction.objects.filter(patient=patient).select_related("test__lab")

    return render(request, "view_prediction_results.html", {"predictions": predictions})

@login_required
def give_feedback(request):
    if request.method == "POST":
        lab_id = request.POST.get("lab_id")
        message = request.POST.get("message")

        lab = get_object_or_404(Lab, id=lab_id)
        patient = request.user.patient  # Get the logged-in patient

        Feedback.objects.create(patient=patient, lab=lab, message=message)
        messages.success(request, "Feedback submitted successfully!")
        return redirect("patient_dash")  # Redirect to predictions page

    # ✅ Fetch Labs that uploaded test results for the patient
    labs = Lab.objects.filter(testdetails__patient=request.user.patient).distinct()

    return render(request, "give_feedback.html", {"labs": labs})

@login_required
def view_feedback(request):
    """Display feedback received by the logged-in lab."""
    if not hasattr(request.user, "lab"):  # Ensure the user is a lab
        return render(request, "error.html", {"message": "You are not authorized to view this page."})

    lab = request.user.lab  # Get the logged-in lab
    feedbacks = Feedback.objects.filter(lab=lab).select_related("patient")  # Fetch feedback for the lab

    return render(request, "view_feedback.html", {"feedbacks": feedbacks})


# ✅ Load the Deep Learning Model (No Database Required)

# import os
# import numpy as np
# import tensorflow as tf
# from django.shortcuts import render
# from django.core.files.storage import FileSystemStorage
# from tensorflow.keras.preprocessing import image

# MODEL_PATH = os.path.join(os.path.dirname(__file__), "ecg_model.keras")

# try:
#     dl_model = tf.keras.models.load_model(MODEL_PATH)
#     print("✅ DL Model Loaded Successfully")
# except Exception as e:
#     print(f"❌ Error Loading Model: {e}")

# # ✅ Function to render image input page
# def image_input(request):
#     return render(request, "image_input.html")

# # ✅ Function to process uploaded image and predict
# def image_result(request):
#     if request.method == "POST":
#         uploaded_file = request.FILES.get("file")
#         if not uploaded_file:
#             return render(request, "image_input.html", {"error": "No file uploaded."})

#         # ✅ Save the uploaded image temporarily
#         fs = FileSystemStorage()
#         file_path = fs.save(uploaded_file.name, uploaded_file)
#         file_url = fs.url(file_path)

#         # ✅ Preprocess the image for prediction
#         img_path = fs.path(file_path)
#         img = image.load_img(img_path, target_size=(224, 224))  # Resize image
#         img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#         # ✅ Make Prediction
#         prediction = dl_model.predict(img_array)
#         predicted_label = "Abnormal" if np.argmax(prediction) == 1 else "Normal"

#         # ✅ Render result page with the uploaded image and prediction
#         return render(request, "image_result.html", {"image_url": file_url, "res": predicted_label})

#     return render(request, "image_input.html")

@login_required
def p_request_test(request):
    labs = Lab.objects.filter(is_approved=True)  # Fetch only approved labs
    return render(request, 'p_request_test.html', {'labs': labs})

@login_required
def submit_test_request(request):
    if request.method == "POST":
        selected_tests = request.POST.getlist('tests')  # Get selected tests as a list
        lab_id = request.POST.get('lab')  # Get selected lab ID
        lab = get_object_or_404(Lab, id=lab_id)
        
        # Retrieve the Patient instance corresponding to the logged-in user
        patient_instance = get_object_or_404(Patient, user=request.user)
        
        # Create the TestRequest with the Patient instance
        test_request = TestRequest.objects.create(
            patient=patient_instance,
            lab=lab,
            requested_tests=", ".join(selected_tests),
            status="Pending"
        )
        return redirect('patient_dash')  # Redirect to the patient dashboard after submission

    return redirect('request_test')

@login_required
def l_approve_requests(request):
    lab_requests = TestRequest.objects.filter(lab=request.user.lab, status="Pending")  # Show only pending requests for the logged-in lab
    return render(request, 'l_approve_requests.html', {'requests': lab_requests})

@login_required
def update_request_status(request, request_id, status):
    test_request = TestRequest.objects.get(id=request_id)
    if test_request.lab == request.user.lab:  # Ensure only assigned lab updates it
        test_request.status = status
        test_request.save()
    return redirect('l_approve_requests')  # Refresh the page

def p_track_requests(request):
    patient_instance = get_object_or_404(Patient, user=request.user)
    patient_requests = TestRequest.objects.filter(patient=patient_instance) # Get requests for logged-in patient
    return render(request, 'p_track_requests.html', {'requests': patient_requests})

@login_required
def p_request_details(request, request_id):
    # Retrieve the Patient instance linked to the logged-in user
    patient_instance = get_object_or_404(Patient, user=request.user)
    
    # Retrieve the test request, ensuring it belongs to the retrieved patient
    test_request = get_object_or_404(TestRequest, id=request_id, patient=patient_instance)
    
    # Retrieve the corresponding test details (if any) for this patient and lab
    test_details = TestDetails.objects.filter(patient=patient_instance, lab=test_request.lab).first()
    
    return render(request, "p_request_details.html", {
        "test_request": test_request,
        "test_details": test_details,
    })

def bill_details(request, prediction_id):
    """
    Generate and display a bill for a given prediction.
    If a bill already exists for the prediction, show it.
    Otherwise, create one (here we set a dummy amount, status, etc.)
    """
    # Ensure the user is a patient
    if not hasattr(request.user, "patient"):
        return render(request, "error.html", {"message": "You are not authorized to view this page."})
    
    prediction = get_object_or_404(Prediction, id=prediction_id, patient=request.user.patient)
    
    # Try to get an existing bill for this prediction
    bill = prediction.bills.first()
    if not bill:
        # Create a new bill (adjust total_amount as needed)
        bill = Bill.objects.create(
            prediction=prediction,
            lab=prediction.test.lab,
            total_amount=500.00,   # Set the proper amount if available
            status="Pending"     # You could mark as 'Issued'
        )
    
    return render(request, "bill_details.html", {"bill": bill})

# views.py



from django.shortcuts import render, get_object_or_404
from django.conf import settings
import razorpay
from APP.models import Bill

def pay_bill(request, bill_id):
    bill = get_object_or_404(Bill, id=bill_id)
    amount_in_rupees = max(bill.total_amount, 1)
    amount_in_paise  = int(amount_in_rupees * 100)

    print(f"[DEBUG] Razorpay Key ID: {settings.RAZORPAY_KEY_ID}")

    client = razorpay.Client(
        auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET)
    )

    order = None
    error = None
    try:
        order = client.order.create({
            "amount": amount_in_paise,
            "currency": "INR",
            "receipt": f"bill_{bill.id}",
            "payment_capture": 1,
        })
        bill.status = "Pending"
        bill.save()
    except razorpay.errors.BadRequestError as e:
        error = str(e)

    return render(request, "pay_bill.html", {
        "order": order,
        "bill": bill,
        "error": error,
        "razorpay_key_id": settings.RAZORPAY_KEY_ID,
    })



from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
def payment_success(request, bill_id):
    if request.method == "POST":
        # Get the bill
        bill = get_object_or_404(Bill, id=bill_id)

        # Update bill status
        bill.status = "Success"
        bill.save()

        # Render success page
        return render(request, 'payment_success.html', {'bill_id': bill_id})
    
    return HttpResponse("Invalid request")