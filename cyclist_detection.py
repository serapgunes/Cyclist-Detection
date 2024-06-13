from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('best.pt')

source = 'video.mp4'
cap = cv2.VideoCapture(source)

small_img_width = 60

circle_mask = np.zeros((small_img_width, small_img_width, 3), dtype=np.uint8)
cv2.circle(circle_mask, (small_img_width // 2, small_img_width // 2), small_img_width // 2, (255, 255, 255), -1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for i, r in enumerate(results):
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            overlay_img = cv2.imread('bisikletlevha.jpg')

            small_img_height = int(overlay_img.shape[0] * (small_img_width / overlay_img.shape[1]))
            #İlk olarak, orijinal resmin yüksekliği overlay_img.shape[0] ve genişliği overlay_img.shape[1] kullanılarak resmin oranı hesaplanır. Ardından, small_img_width ile bu oran çarpılır ve sonuç yuvarlanarak tam sayıya dönüştürülür.
            # Bu işlem, orijinal resmin oranını koruyarak, small_img_width genişliğindeki resmin yüksekliğini belirlemek için yapılır.
            #------------------
            # Daha sonra, cv2.resize fonksiyonu kullanılarak orijinal resim overlay_img,
            # belirlenen genişlik ve yükseklik değerlerine (small_img_width ve small_img_height) göre boyutlandırılır. Bu işlem, belirlenen boyutlarda bir resim elde etmek için yapılır.
            # Bu boyutlandırılmış resim, algılanan nesnenin üst kısmına eklenecek olan resimdir.
            small_overlay_img = cv2.resize(overlay_img, (small_img_width, small_img_height))# overlay_img'yi küçük boyuta dönüştürür.
            # Bu, algılanan nesnenin üst kısmına eklenmek üzere boyutlandırılmış bir resimdir.

            # Sınırlayıcı kutunun tepe noktasını bul. Bu işlem, eklenmek istenen resmin nesne üzerindeki konumunu belirler.
            # Özellikle, nesnenin sınırlayıcı kutusunun üst kenarına eklenmek istenen resmin konumunu belirler.
            # y1: Algılanan nesnenin sınırlayıcı kutusunun sol üst köşesinin y koordinatı.
            # small_img_height: Eklenmek istenen resmin yüksekliği.
            # Bu iki değer arasındaki fark, nesnenin sınırlayıcı kutusunun yüksekliğinden eklenmek istenen resmin yüksekliğini çıkarır.
            # Sonuç, eklenmek istenen resmin nesnenin üst kenarından ne kadar yukarıda başlaması gerektiğini belirler.
            # Ancak, bu hesaplama negatif bir değer verirse, yani eklenmek istenen resim nesnenin üst kenarının üstünde konumlanıyorsa,
            # max(0, ...) işlemi bu değeri 0 ile sınırlar. Bu, eğer eklenmek istenen resim nesnenin üstünden taşarsa,
            # resmin ekranın üst kenarına sıfır piksel mesafede yerleştirileceği anlamına gelir.
            # Sonuç olarak, top_position değişkeni, eklenmek istenen resmin nesnenin üst kenarına olan dikey konumunu belirler.
            top_position = max(0, y1 - small_img_height)

            # Sınırlayıcı kutunun genişliğinin yarısı kadar sola kaydır
            x_position = max(0, x1 + int((x2 - x1 - small_img_width) / 2))

            # Küçük boyutlu resmi sınırlayıcı kutunun tepe noktasına yerleştir
            overlay_area = frame[top_position:top_position + small_img_height, x_position:x_position + small_img_width]

            # Daire maskesini büyüt
            resized_circle_mask = cv2.resize(circle_mask, (overlay_area.shape[1], overlay_area.shape[0]))

            # Daire maskesini uygula
            masked_overlay = cv2.bitwise_and(small_overlay_img, resized_circle_mask)
            masked_frame = cv2.bitwise_and(overlay_area, cv2.bitwise_not(resized_circle_mask))

            # Son olarak, maskelenmiş resimleri ekleyin
            frame[top_position:top_position + small_img_height, x_position:x_position + small_img_width] = cv2.add(masked_overlay, masked_frame)
            #Maskeleme işleminden sonra, maske alanını karedeki ilgili bölgeyle birleştirir.

    cv2.imshow('Detected Objects', frame)#İşlenmiş kareyi görselleştirir.

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
