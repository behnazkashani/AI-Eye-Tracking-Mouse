import cv2
import pyautogui as robot
import numpy as np
import time


def main():

    # مدل‌ها
    eye_model = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    face_model = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


    # برای نرم کردن حرکت موس
    prev_x = 0
    prev_y = 0

    # جلوگیری از کلیک پشت سر هم
    last_click = 0


    cam = cv2.VideoCapture(0)


    while True:

        ret, img = cam.read()

        if not ret:
            continue


        img = cv2.flip(img, 1)

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )


        imgout = img.copy()


        # تشخیص چهره
        face = face_model.detectMultiScale(gray)


        if len(face) > 0:

            xs = face[0][0]
            ys = face[0][1]

            w = face[0][2]
            h = face[0][3]

            xs2 = xs + w
            ys2 = ys + h


            # رسم کادر چهره
            cv2.rectangle(
                imgout,
                (xs, ys),
                (xs2, ys2),
                (0,255,0),
                3
            )


            # مرکز صورت
            center_x = xs + w // 2
            center_y = ys + h // 2


            screen_width, screen_height = robot.size()


            move_x = np.interp(
                center_x,
                [0, img.shape[1]],
                [0, screen_width]
            )

            move_y = np.interp(
                center_y,
                [0, img.shape[0]],
                [0, screen_height]
            )


            # Smooth حرکت موس
            smooth_x = prev_x + (move_x - prev_x) * 0.2
            smooth_y = prev_y + (move_y - prev_y) * 0.2


            robot.moveTo(
                smooth_x,
                smooth_y,
                duration=0.05
            )


            prev_x = smooth_x
            prev_y = smooth_y



            # محدوده چشم
            gray_face = gray[
                ys:ys2,
                xs:xs2
            ]


            eye = eye_model.detectMultiScale(
                gray_face,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30,30)
            )


            # رسم چشم‌ها
            count = 0

            for (xe, ye, ew, eh) in eye:

                count += 1

                cv2.rectangle(
                    imgout,
                    (xe+xs, ye+ys),
                    (xe+xs+ew, ye+ys+eh),
                    (255,0,0),
                    2
                )

                if count == 2:
                    break



            # تشخیص بسته شدن چشم
            if len(eye) == 0:

                current_time = time.time()

                # هر یک ثانیه فقط یک کلیک
                if current_time - last_click > 1:

                    robot.click()

                    last_click = current_time

                    print("Click!")



        cv2.imshow(
            "AI Eye Tracking Mouse",
            imgout
        )


        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break



    cam.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()