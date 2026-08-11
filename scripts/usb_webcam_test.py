import cv2


def main() -> None:
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Cannot open USB webcam at index 1")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Cannot read frame from USB webcam")
            cv2.imshow("USB Webcam Test - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
