import os
import numpy as np
import cv2
import dlib
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from PIL import Image
from .models import Users
from scipy.spatial import distance


# Create your views here.

def auth(request):
    if request.method == 'POST':
        image = request.FILES['image']
        fs = FileSystemStorage()
        fs.save(image.name, image)
        best_match, best_distance = compare_faces(image.name)
        user_fio = best_match

        # Заданная точность не больше 0.6 для фронтальных изображений лица
        if best_distance < 0.6:
            return render(request, 'main/source_page.html', {'user_fio': user_fio})

    return render(request, 'main/auth.html')


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('C:/Users/Admin/PycharmProjects/App/authface/main/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1(
    'C:/Users/Admin/PycharmProjects/App/authface/main/dlib_face_recognition_resnet_model_v1.dat')


def compare_faces(image_path, threshold=0.6):
    # Загружаем изображение из базы данных
    user_photos = Users.objects.all()

    # Загружаем и обрабатываем изображение для сравнения
    img = dlib.load_rgb_image(image_path)
    dets = detector(img, 1)

    # Если лицо не обнаружено, вернуть None
    if len(dets) == 0:
        return None

    best_match = None
    best_distance = threshold
    # Проход по всем изображениям из бд
    for user_photo in user_photos:
        user_img = dlib.load_rgb_image(user_photo.user_photo.path)

        user_dets = detector(user_img, 1)
        if len(user_dets) == 0:
            continue

        user_shape = sp(user_img, user_dets[0])
        user_face_descriptor = facerec.compute_face_descriptor(user_img, user_shape)

        for det in dets:
            shape = sp(img, det)
            face_descriptor = facerec.compute_face_descriptor(img, shape)

            # Вычисление Евклидова расстояния между исходным изображением и фото профиля из бд
            distance = np.linalg.norm(np.array(face_descriptor) - np.array(user_face_descriptor))
            print(distance, user_photo.user_fio)

            # Вычисление наилучшего найденного результата
            if distance < best_distance:
                best_distance = distance
                best_match = user_photo.user_fio
    return best_match, best_distance


