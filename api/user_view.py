from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from rest_framework import status
import json
from .models import Student, Career,Semester
from .serializer import StudentSerializer

@require_http_methods(["GET"])
def get_student_by_registration(request, registration):
    
    
    try:
        student = Student.objects.get(registration_student= registration)
        
        studentSerializer = StudentSerializer(student)
        return JsonResponse({'data': [studentSerializer.data]}, status=status.HTTP_200_OK)
        
        
    except Student.DoesNotExist:
        return JsonResponse({'error': 'User not found', 'data': []}, status=status.HTTP_404_NOT_FOUND)
   

@csrf_exempt
def create_user(request):
    if request.method == 'POST':
        
        try:
            data = json.loads(request.body)
       
            name_student, registration_student, semester_id, career_id = (
                data.get(key) for key in ('name_student', 'registration_student', 'semester_id', 'career_id')
            )
            # print(name_student, registration_student, semester_id, career_id)

            if not all([name_student, registration_student, semester_id, career_id]):
                return JsonResponse({'status': 'error', 'message': 'Missing fields'}, status=400)

            semester = Semester.objects.get(pk=int(semester_id))
            career = Career.objects.get(pk=int(career_id))
            
   
            student = Student(
                name_student=name_student,
                registration_student=registration_student,
                semester=semester,
                career=career
            )
            # print(student)
            student.save()
            
            student_serializer = StudentSerializer(student)
            
            return JsonResponse({'data':[student_serializer.data]})
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON format'}, status=400)
        
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})
  
