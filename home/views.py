from django.http import HttpResponse
from django.shortcuts import render

from .models import Image

# from .tweet_predictor import predict
from .predict import predict

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

        result = predict(image.img.name)
        result = result[0]
        print(result)

        context = {
            'image': image,
            'result': result,
        }
        return render(request, 'home/index.html', context)

    return render(request, 'home/index.html')
