

# from django.contrib import admin
# from django.urls import path
# from .views import CreateAlertView

# urlpatterns = [
#     path('admin/', admin.site.urls),
    
#     path('api/alerts/create/', CreateAlertView.as_view(), name='create-alert'),
    
# ]






from django.contrib import admin
from django.http import JsonResponse
from django.urls import path
from .views import CreateAlertView

def root_view(request):
    return JsonResponse({"message": "API is running. Use POST requests."})

urlpatterns = [
    path('', root_view),
    path('admin/', admin.site.urls),
    path('api/alerts/create/', CreateAlertView.as_view(), name='create-alert'),
]
