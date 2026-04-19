from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from model.predict import predict_video

def home(request):
    if request.method == 'POST':
        video = request.FILES.get('video')
        frame_count = int(request.POST.get('frames'))

        fs = FileSystemStorage()
        filename = fs.save(video.name, video)
        filepath = fs.path(filename)

        result = predict_video(filepath, frame_count)

        return render(request, 'index.html', {
            'results': result['models'],
            'final': result['final'],
            'confidence': result['confidence'],
            'video_url': fs.url(filename),
            'frames': frame_count
        })

    return render(request, 'index.html')