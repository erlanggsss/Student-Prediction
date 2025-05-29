import datetime
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import io

buffer = io.BytesIO()

# Define function for data preprocessing
def data_preprocessing(data_input, single_data, n):
    df = pd.read_csv('./data/student_data_filtered.csv')  # Assuming the file path is correct
    df = df.drop(columns=['Status'], axis=1)
    df = pd.concat([data_input, df])
    df = StandardScaler().fit_transform(df)

    return df[[n]] if single_data else df[0 : n]

# Define function to load and use model for prediction
def model_predict(df):
    model = joblib.load('./models/rf_model.pkl')  # Assuming the model is saved correctly
    return model.predict(df)

# Define a function to apply color mapping for predictions
def color_mapping(value):
    color = 'green' if value == 'Lulus' else 'red'
    return f'color: {color}'

# Streamlit App Configuration
st.set_page_config(
    page_title="Dashboard Prediksi Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

st.title('🎓 Dashboard Prediksi Status Mahasiswa')
st.markdown("---")

with st.expander("📚 Penjelasan Jalur Pendaftaran"):
    st.markdown("""
    **Jalur Pendaftaran** menjelaskan cara mahasiswa mendaftar ke institusi. Berikut adalah penjelasan untuk setiap kategori:
    
    **🎯 Fase Umum:**
    - **1st Phase - General Contingent**: Pendaftaran fase pertama untuk kuota umum
    - **2nd Phase - General Contingent**: Pendaftaran fase kedua untuk kuota umum  
    - **3rd Phase - General Contingent**: Pendaftaran fase ketiga untuk kuota umum
    
    **🏝️ Kuota Khusus Regional:**
    - **1st Phase - Special Contingent (Azores Island)**: Kuota khusus untuk penduduk Kepulauan Azores
    - **1st Phase - Special Contingent (Madeira Island)**: Kuota khusus untuk penduduk Pulau Madeira
    
    **📋 Berdasarkan Peraturan:**
    - **Ordinance No. 612/93**: Pendaftaran berdasarkan Peraturan No. 612/93
    - **Ordinance No. 854-B/99**: Pendaftaran berdasarkan Peraturan No. 854-B/99
    - **Ordinance No. 533-A/99, Item B2**: Pendaftaran dengan rencana studi berbeda
    - **Ordinance No. 533-A/99, Item B3**: Pendaftaran dari institusi lain
    
    **🌍 Mahasiswa Internasional:**
    - **International Student (Bachelor)**: Mahasiswa internasional program sarjana
    
    **👥 Kategori Khusus:**
    - **Over 23 Years Old**: Pendaftar berusia di atas 23 tahun
    - **Transfer**: Mahasiswa pindahan dari institusi lain
    - **Change of Course**: Mahasiswa yang pindah program studi
    - **Change of Institution/Course**: Pindah institusi dan program studi
    - **Change of Institution/Course (International)**: Pindah institusi dan program studi (internasional)
    
    **🎓 Berdasarkan Kualifikasi:**
    - **Holders of Other Higher Courses**: Pemegang gelar pendidikan tinggi lain
    - **Short Cycle Diploma Holders**: Pemegang diploma siklus pendek
    - **Technological Specialization Diploma Holders**: Pemegang diploma spesialisasi teknologi
    """)
    
    st.info("💡 **Tips**: Mode aplikasi ini mempengaruhi proses seleksi dan persyaratan yang harus dipenuhi mahasiswa.")
st.info("Silahkan isi semua data untuk melakukan prediksi status mahasiswa.")

# Mapping categorical data
gender_mapping = {'Pria': 1, 'Wanita': 0}
marital_status_mapping = {
    'Lajang': 1, 'Menikah': 2, 'Janda/Duda': 3, 
    'Bercerai': 4, 'Kohabitasi': 5, 'Berpisah Secara Hukum': 6
}
application_mapping = {
    '1st Phase - General Contingent': 1, 
    '1st Phase - Special Contingent (Azores Island)': 5, 
    '1st Phase - Special Contingent (Madeira Island)': 16, 
    '2nd Phase - General Contingent': 17,
    '3rd Phase - General Contingent': 18, 
    'Ordinance No. 612/93': 2, 
    'Ordinance No. 854-B/99': 10,
    'Ordinance No. 533-A/99, Item B2 (Different Plan)': 26, 
    'Ordinance No. 533-A/99, Item B3 (Other Institution)': 27,
    'International Student (Bachelor)': 15, 
    'Over 23 Years Old': 39, 
    'Transfer': 42, 
    'Change of Course': 43,
    'Holders of Other Higher Courses': 7, 
    'Short Cycle Diploma Holders': 53,
    'Technological Specialization Diploma Holders': 44, 
    'Change of Institution/Course': 51,
    'Change of Institution/Course (International)': 57,
}

# Initialize session state for form data
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

# Create form for data input
with st.form(key='student_data_form'):
    st.subheader('📋 Informasi Mahasiswa')
    
    # Row 1: Personal Information
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        gender = st.radio(
            'Jenis Kelamin', 
            options=['Pria', 'Wanita'], 
            help='Pilih jenis kelamin mahasiswa'
        )
    
    with col2:
        age = st.number_input(
            'Umur saat Pendaftaran', 
            min_value=17, 
            max_value=70, 
            value=22,
            help='Umur pada saat mendaftar (17-70)'
        )
    
    with col3:
        marital_status = st.selectbox(
            'Status Pernikahan', 
            options=['Lajang', 'Menikah', 'Duda', 'Bercerai', 'Kumpul Kebo', 'Berpisah Secara Hukum'],
            help='Pilih status pernikahan'
        )
    
    st.markdown("---")
    
    # Row 2: Academic Information
    st.subheader('🎓 Informasi Akademik')
    
    col4, col5, col6 = st.columns([2, 1, 1])
    
    with col4:
        application_mode = st.selectbox(
            'Jalur Pendaftaran', 
            options=list(application_mapping.keys()), 
            help='Pilih jalur pendaftaran'
        )
    
    with col5:
        prev_qualification_grade = st.number_input(
            'Nilai Kualifikasi Sebelumnya', 
            min_value=0, 
            max_value=200, 
            value=86,
            help='Nilai kualifikasi sebelumnya (0-200)'
        )
    
    with col6:
        admission_grade = st.number_input(
            'Nilai Penerimaan', 
            min_value=0, 
            max_value=200, 
            value=93,
            help='Nilai penerimaan (0-200)'
        )
    
    st.markdown("---")
    
    # Row 3: Financial & Status Information
    st.subheader('💰 Informasi Finansial & Status')
    
    col7, col8, col9, col10 = st.columns(4)
    
    with col7:
        scholarship_holder = st.checkbox(
            'Penerima Beasiswa', 
            value=True,
            help='Centang jika mahasiswa penerima beasiswa'
        )
    
    with col8:
        tuition_fees_up_to_date = st.checkbox(
            'SPP Terbayar', 
            value=True,
            help='Centang jika SPP lunas'
        )
    
    with col9:
        displaced = st.checkbox(
            'Mahasiswa Pindahan', 
            value=False,
            help='Centang jika mahasiswa pindahan'
        )
    
    with col10:
        debtor = st.checkbox(
            'Status Debitur', 
            value=False,
            help='Centang jika mahasiswa memiliki hutang'
        )
    
    st.markdown("---")
    
    # Row 4: Curricular Units Information
    st.subheader('📚 Informasi Akademik')
    
    # Container untuk styling yang lebih rapi
    with st.container():
        # First row of curricular units - Enrolled & Evaluations
        col11, col12, col13 = st.columns(3)
        
        with col11:
            curricular_units_1st_sem_enrolled = st.number_input(
                'Mata Kuliah Semester 1 (Terdaftar)', 
                min_value=0, 
                max_value=26, 
                value=18,
                help='Jumlah mata kuliah yang diambil mahasiswa di semester pertama (0-26)'
            )
        
        with col12:
            curricular_units_2nd_sem_enrolled = st.number_input(
                'Mata Kuliah Semester 2 (Terdaftar)', 
                min_value=0, 
                max_value=23, 
                value=20,
                help='Jumlah mata kuliah yang diambil mahasiswa di semester kedua (0-23)'
            )
        
        with col13:
            curricular_units_1st_sem_evaluations = st.number_input(
                'Mata Kuliah Semester 1 (Tidak Lulus)', 
                min_value=0, 
                max_value=33, 
                value=0,
                help='Jumlah evaluasi mata kuliah mahasiswa di semester kedua (0-33)'
            )
    
    # Spacing untuk pemisahan visual
    st.write("")
    
    with st.container():
        # Second row of curricular units - Approved
        col14, col15, col16 = st.columns(3)
        
        with col14:
            curricular_units_1st_sem_approved = st.number_input(
                'Mata Kuliah Semester 1 (Lulus)', 
                min_value=0, 
                max_value=26, 
                value=18,
                help='Jumlah mata kuliah yang lulus di semester pertama (0-26)'
            )
        
        with col15:
            curricular_units_2nd_sem_approved = st.number_input(
                'Mata Kuliah Semester 2 (Lulus)', 
                min_value=0, 
                max_value=20, 
                value=19,
                help='Jumlah mata kuliah yang lulus di semester kesatu (0-20)'
            )

        with col16:
            curricular_units_2nd_sem_evaluations = st.number_input(
                'Mata Kuliah Semester 2 (Tidak Lulus)', 
                min_value=0, 
                max_value=33, 
                value=1,
                help='Jumlah evaluasi mata kuliah mahasiswa di semester kedua (0-33)'
            )
    
    # Spacing untuk pemisahan visual
    st.write("")
    
    with st.container():
        # Third row of curricular units - Grades
        col17, col18, col19 = st.columns([1, 1, 1])
        
        with col17:
            curricular_units_1st_sem_grade = st.number_input(
                'Nilai Rata-Rata Semester 1', 
                min_value=0.0, 
                max_value=20.0, 
                value=18.0,
                step=0.1,
                help='Nilai rata-rata semester pertama (0.0-20.0)'
            )
        
        with col18:
            curricular_units_2nd_sem_grade = st.number_input(
                'Nilai Rata-Rata Semester 2', 
                min_value=0.0, 
                max_value=20.0, 
                value=18.0,
                step=0.1,
                help='Nilai rata-rata semester kedua (0.0-20.0)'
            )
        
        with col19:
            st.empty()  # Empty column for alignment
    
    # Form submit button dengan styling yang lebih menarik
    st.markdown("---")
    form_submitted = st.form_submit_button('📝 Simpan Data',  use_container_width=True)
    
    if form_submitted:
        # Store form data in session state
        st.session_state.form_data = {
            'gender': gender,
            'age': age,
            'marital_status': marital_status,
            'application_mode': application_mode,
            'prev_qualification_grade': prev_qualification_grade,
            'admission_grade': admission_grade,
            'scholarship_holder': scholarship_holder,
            'tuition_fees_up_to_date': tuition_fees_up_to_date,
            'displaced': displaced,
            'debtor': debtor,
            'curricular_units_1st_sem_enrolled': curricular_units_1st_sem_enrolled,
            'curricular_units_2nd_sem_enrolled': curricular_units_2nd_sem_enrolled,
            'curricular_units_1st_sem_evaluations': curricular_units_1st_sem_evaluations,
            'curricular_units_2nd_sem_evaluations': curricular_units_2nd_sem_evaluations,
            'curricular_units_1st_sem_approved': curricular_units_1st_sem_approved,
            'curricular_units_2nd_sem_approved': curricular_units_2nd_sem_approved,
            'curricular_units_1st_sem_grade': curricular_units_1st_sem_grade,
            'curricular_units_2nd_sem_grade': curricular_units_2nd_sem_grade
        }
        st.success('✅ Data berhasil disimpan!')

# Prediction Section (Outside the form)
st.markdown("---")
st.subheader('🔮 Bagian Prediksi')

# Create columns for prediction button and result
pred_col1, pred_col2 = st.columns([1, 2])

with pred_col1:
    predict_button = st.button(
        '🚀 Prediksi Status Mahasiswa', 
        type='primary',
        use_container_width=True,
        disabled=len(st.session_state.form_data) == 0
    )

with pred_col2:
    if len(st.session_state.form_data) == 0:
        st.info('Silakan isi dan simpan data formulir terlebih dahulu sebelum melakukan prediksi.')

# Perform prediction when button is clicked
if predict_button and len(st.session_state.form_data) > 0:
    try:
        # Get data from session state
        form_data = st.session_state.form_data
        
        # Map categorical data
        gender_mapped = gender_mapping.get(form_data['gender'])
        marital_status_mapped = marital_status_mapping.get(form_data['marital_status'])
        application_mode_mapped = application_mapping.get(form_data['application_mode'])
        
        # Convert boolean to int
        scholarship_holder_int = 1 if form_data['scholarship_holder'] else 0
        tuition_fees_up_to_date_int = 1 if form_data['tuition_fees_up_to_date'] else 0
        displaced_int = 1 if form_data['displaced'] else 0
        debtor_int = 1 if form_data['debtor'] else 0
        
        # Create data array for prediction
        data = [[
            marital_status_mapped, 
            application_mode_mapped, 
            form_data['prev_qualification_grade'], 
            form_data['admission_grade'], 
            displaced_int, 
            debtor_int, 
            tuition_fees_up_to_date_int,
            gender_mapped, 
            scholarship_holder_int, 
            form_data['age'], 
            form_data['curricular_units_1st_sem_enrolled'],
            form_data['curricular_units_1st_sem_approved'], 
            form_data['curricular_units_1st_sem_grade'],
            form_data['curricular_units_1st_sem_enrolled'], 
            form_data['curricular_units_2nd_sem_evaluations'],
            form_data['curricular_units_2nd_sem_evaluations'],
            form_data['curricular_units_2nd_sem_approved'], 
            form_data['curricular_units_2nd_sem_grade']
        ]]

        # Create DataFrame
        df = pd.DataFrame(data, columns=[
            'Marital_status', 'Application_mode', 'Previous_qualification_grade', 'Admission_grade', 
            'Displaced', 'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 
            'Age_at_enrollment', 'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_approved', 
            'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_enrolled', 
            'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved', 
            'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations'
        ])

        # Preprocess data and make prediction
        data_input = data_preprocessing(df, True, 0)
        output = model_predict(data_input)
        prediction = 'Graduate' if output[0] == 1 else 'Dropout'
        
        # Display prediction result with styling
        st.markdown("### 📊 Hasil Prediksi")
        
        if prediction == 'Graduate':
            st.success(f'🎉 **Prediksi: {prediction}**')
            st.balloons()
        else:
            st.error(f'⚠️ **Prediksi: {prediction}**')
        
        # Display prediction confidence or additional info
        if prediction == 'Graduate':
            st.info(f'Berdasarkan informasi mahasiswa yang diberikan, model memprediksi bahwa mahasiswa kemungkinan akan **{prediction.lower()}**.')
        else:
            st.warning(f'Berdasarkan informasi mahasiswa yang diberikan, model memprediksi bahwa mahasiswa kemungkinan akan **{prediction.lower()}**.')

        
    except Exception as e:
        st.error(f'❌ Terjadi kesalahan saat prediksi: {str(e)}')
        st.error('Pastikan semua file data dan model yang diperlukan tersedia.')

# Display current form data (for debugging/verification)
if len(st.session_state.form_data) > 0:
    with st.expander('📋 Lihat Data Formulir Saat Ini'):
        st.json(st.session_state.form_data)

# Footer
st.markdown("---")
year_now = datetime.date.today().year
year = year_now if year_now == 2025 else f'2025 - {year_now}'
name = "Muhammad Erlangga Prasetya"
copyright = f'Copyright © {year} {name}'
st.caption(copyright)