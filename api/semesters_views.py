from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from rest_framework import status
    
from .serializer import SemesterSerializer
from .models import Semester

@require_http_methods(["GET"])
def get_semesters(request):
    
    semesters = Semester.objects.all()
    semestersSerializer = SemesterSerializer(semesters, many=True)

    return JsonResponse({'status': status.HTTP_200_OK, 'data' : semestersSerializer.data})
  
