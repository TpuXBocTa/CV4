import cv2


def select_roi_on_frame(window_name, frame):
    roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    if roi == (0, 0, 0, 0):
        return None
    return roi


def create_tracker(method_name: str):
    if method_name == "CSRT":
        return cv2.legacy.TrackerCSRT_create()
    elif method_name == "KCF":
        return cv2.legacy.TrackerKCF_create()
    elif method_name == "MOSSE":
        return cv2.legacy.TrackerMOSSE_create()
    else:
        raise ValueError(f"Unknown tracker method: {method_name}")


def preprocess_frame(frame, method_name: str):
    if method_name != "MOSSE":
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return processed


def main():
    cap = cv2.VideoCapture("video.mov")

    window_name = "Multi-tracker (CSRT/KCF/MOSSE) – q: quit, 1/2/3: change method"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    ret, frame = cap.read()
    if not ret:
        print("Failed to read first frame.")
        cap.release()
        cv2.destroyAllWindows()
        return

    current_method = "CSRT"
    print("Available tracking methods:")
    print("1 – CSRT (default)")
    print("2 – KCF")
    print("3 – MOSSE (with preprocessing)")
    print(f"Current method: {current_method}")
    print("Select object and press ENTER/SPACE. ESC – cancel.")

    roi = select_roi_on_frame(window_name, frame)
    if roi is None:
        print("ROI not selected. Quitting...")
        cap.release()
        cv2.destroyAllWindows()
        return

    tracker = create_tracker(current_method)
    processed_for_tracker = preprocess_frame(frame, current_method)
    tracker.init(processed_for_tracker, roi)

    quitting = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break

        processed_for_tracker = preprocess_frame(frame, current_method)

        success, box = tracker.update(processed_for_tracker)

        if success:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(
                frame,
                f"{current_method}: Tracking",
                (x, max(y - 15, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,  # bigger font
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                f"{current_method}: Tracking lost - select new target or press q to quit",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,  # bigger font
                (0, 0, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Method: {current_method}  |  1-CSRT, 2-KCF, 3-MOSSE, ENTER - new ROI, q/ESC - exit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            while True:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(0) & 0xFF

                if key == ord("q") or key == 27:
                    quitting = True
                    break

                elif key == 13:
                    print(f"{current_method}: select new object and press ENTER/SPACE. ESC – cancel.")
                    roi = select_roi_on_frame(window_name, frame)
                    if roi is not None:
                        tracker = create_tracker(current_method)
                        processed_for_tracker = preprocess_frame(frame, current_method)
                        tracker.init(processed_for_tracker, roi)
                    else:
                        print("ROI not selected. Continuing without changes...")
                    break

                elif key in (ord("1"), ord("2"), ord("3")):
                    if key == ord("1"):
                        new_method = "CSRT"
                    elif key == ord("2"):
                        new_method = "KCF"
                    else:
                        new_method = "MOSSE"

                    if new_method != current_method:
                        current_method = new_method
                        print(f"Switched method to {current_method}. Select new object.")
                        roi = select_roi_on_frame(window_name, frame)
                        if roi is not None:
                            tracker = create_tracker(current_method)
                            processed_for_tracker = preprocess_frame(frame, current_method)
                            tracker.init(processed_for_tracker, roi)
                        else:
                            print("ROI not selected. Keeping previous tracker (if still valid).")
                    break

            if quitting:
                break

            continue

        cv2.putText(
            frame,
            f"Method: {current_method}  |  1-CSRT, 2-KCF, 3-MOSSE, ENTER - new ROI, q/ESC - exit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:  # q or ESC
            break

        elif key == 13:
            print(f"{current_method}: select new object and press ENTER/SPACE. ESC – cancel.")
            roi = select_roi_on_frame(window_name, frame)
            if roi is not None:
                tracker = create_tracker(current_method)
                processed_for_tracker = preprocess_frame(frame, current_method)
                tracker.init(processed_for_tracker, roi)
            else:
                print("ROI not selected. Keeping previous tracker...")

        elif key in (ord("1"), ord("2"), ord("3")):
            if key == ord("1"):
                new_method = "CSRT"
            elif key == ord("2"):
                new_method = "KCF"
            else:
                new_method = "MOSSE"

            if new_method != current_method:
                current_method = new_method
                print(f"Switched method to {current_method}. Select new object.")
                roi = select_roi_on_frame(window_name, frame)
                if roi is not None:
                    tracker = create_tracker(current_method)
                    processed_for_tracker = preprocess_frame(frame, current_method)
                    tracker.init(processed_for_tracker, roi)
                else:
                    print("ROI not selected. Keeping previous tracker (if still valid).")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
