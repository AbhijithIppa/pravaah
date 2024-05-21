import streamlit as st

# Set the page configuration
st.set_page_config(page_title="Interview Process", layout="centered")

# Define the sections as functions to navigate to
def resume_shortlisting():
    st.title("Resume Shortlisting")
    st.write("This is the Resume Shortlisting section.")
    # Add your resume shortlisting code here

def technical_interview():
    st.title("Technical Interview")
    st.write("This is the Technical Interview section.")
    # Add your technical interview code here

def hr_interview():
    st.title("HR Interview")
    st.write("This is the HR Interview section.")
    # Add your HR interview code here

# Create a function for the homepage
def homepage():
    st.title("Welcome to the Interview Process")
    st.write("Please select a section to proceed:")
    
    # Add spacing for a better layout
    st.markdown("##")
    
    # Arrange sections using columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("https://cdn.pixabay.com/photo/2017/07/31/11/31/town-2550234_960_720.jpg", 
                 caption='Resume Shortlisting',
                 use_column_width=True)  # Resume Shortlisting Image
        if st.button("Go to Resume Shortlisting"):
            st.session_state.page = "resume_shortlisting"
    
    with col2:
        st.image("https://cdn.pixabay.com/photo/2018/05/30/17/39/interview-3440768_960_720.jpg", 
                 caption='Technical Interview',
                 use_column_width=True)  # Technical Interview Image
        if st.button("Go to Technical Interview"):
            st.session_state.page = "technical_interview"
    
    with col3:
        st.image("https://cdn.pixabay.com/photo/2016/03/23/08/34/time-management-1274699_960_720.jpg", 
                 caption='HR Interview',
                 use_column_width=True)  # HR Interview Image
        if st.button("Go to HR Interview"):
            st.session_state.page = "hr_interview"

# Initialize the session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = "homepage"

# Navigate to the selected page
if st.session_state.page == "homepage":
    homepage()
elif st.session_state.page == "resume_shortlisting":
    resume_shortlisting()
elif st.session_state.page == "technical_interview":
    technical_interview()
elif st.session_state.page == "hr_interview":
    hr_interview()

# Add a back button to navigate back to the homepage
if st.session_state.page != "homepage":
    if st.button("Back to Homepage"):
        st.session_state.page = "homepage"
        st.experimental_rerun()

# Hyperlink navigation
st.markdown(
    """
    <style>
    .nav-link {
        text-decoration: none;
        color: blue;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Define navigation through hyperlinks
if st.session_state.page == "resume_shortlisting":
    st.markdown('<a href="/" class="nav-link" onclick="window.location.reload()">Go to Technical Interview</a>', unsafe_allow_html=True)
    st.markdown('<a href="/" class="nav-link" onclick="window.location.reload()">Go to HR Interview</a>', unsafe_allow_html=True)
elif st.session_state.page == "technical_interview":
    st.markdown('<a href="/" class="nav-link" onclick="window.location.reload()">Go to Resume Shortlisting</a>', unsafe_allow_html=True)
    st.markdown('<a href="/" class="nav-link" onclick="window.location.reload()">Go to HR Interview</a>', unsafe_allow_html=True)
elif st.session_state.page == "hr_interview":
    st.markdown('<a href="/" class="nav-link" onclick="window.location.reload()">Go to Resume Shortlisting</a>', unsafe_allow_html=True)
    st.markdown('<a href="/" class="nav-link" onclick="window.location.reload()">Go to Technical Interview</a>', unsafe_allow_html=True)
