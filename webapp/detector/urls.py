# from django.contrib import admin
# from django.http import JsonResponse
# from django.urls import path
# from .views import CreateAlertView

# def root_view(request):
#     return JsonResponse({"message": "API is running. Use POST requests."})

# urlpatterns = [
#     path('', root_view),
#     path('admin/', admin.site.urls),
#     path('api/alerts/create/', CreateAlertView.as_view(), name='create-alert'),
# ]





# detector/urls.py

from django.urls import path
from .views import dashboard, alert_list_partial, alert_detail, CreateAlertView

urlpatterns = [
    # Frontend URLs
    # detector/urls.py
    path('dashboard/', dashboard, name='dashboard'),
    path('alerts/partial/', alert_list_partial, name='alert-list-partial'),
    path('alert/<int:pk>/', alert_detail, name='alert-detail'), # Add this line

    # API URL
    path('api/alerts/create/', CreateAlertView.as_view(), name='create-alert'),
]