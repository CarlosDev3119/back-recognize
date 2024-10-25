from django.urls import path
from .views import emotion_detection
from .careers_views import get_careers
from .semesters_views import get_semesters
from .user_view import get_student_by_registration, create_user



urlpatterns = [

    path('emotions/recognize/', emotion_detection, name='recognize_emotion'),
    path('career/', get_careers, name='get_careers'),
    path('semester/', get_semesters, name='get_semesters'),
    path('user/<str:registration>/', get_student_by_registration, name='get_student_by_registration'),
    path('user/register/create/', create_user, name='create_user'),
    
]


