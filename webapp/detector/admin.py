from django.contrib import admin

# Register your models here.
# alerts/admin.py
from .models import Alert

# Register your models here.
admin.site.register(Alert)