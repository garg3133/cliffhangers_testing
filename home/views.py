from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render

import png
import matplotlib

from .models import Image

# from .tweet_predictor import predict
# from .predict import predict
from .files.object_detection_tutorial_final import detectObjectFromPathList

# Create your views here.
def index(request):
    if request.POST:
        tweet = request.POST['tweet']

        # result = predict(tweet)
        # score = result[0][0]
        # if score > 0.5:
        #     analysis = 'Positive'
        # else:
        #     analysis = 'Negative'

        # context = {
        #     'tweet': tweet,
        #     'score': score,
        #     'analysis': analysis,
        # }
        # return render(request, 'home/index.html', context)

    return render(request, 'home/index.html')

def imageClassifier(request):
    if not Image.objects.all().exists():
        Image.objects.create()

    if request.POST:
        img = request.FILES['img']
        image = Image.objects.all()[0]
        image.img.delete(False)
        image.img = img
        image.save()

        paths = [settings.MEDIA_ROOT + '\\' + image.img.name,]
        print(paths)
        result = detectObjectFromPathList(paths)
        result = result[0]

        path = settings.MEDIA_ROOT + '\\result.png'
        matplotlib.image.imsave(path, result)
        Image.objects.filter(pk=1).update(res='result.png')
        image = Image.objects.get(pk=1)
        # print(image.res, image.res.name, image.res.url, image.img, image.img.name, image.img.url )
        # print(result)

        context = {
            'image': image,
            'result': result,
            'show': True,
        }
        return render(request, 'home/index.html', context)

    return render(request, 'home/index.html')
