from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
import os
urlpatterns = [
    path('', views.index, name='index'),  # NEW — root route
    path('index.html', views.index),       # optional
    path('tiktok/', views.tiktok_audio_analysis, name='tiktok_audio_analysis'),
]

# Serve static files in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=os.path.join(settings.BASE_DIR, 'static'))
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)