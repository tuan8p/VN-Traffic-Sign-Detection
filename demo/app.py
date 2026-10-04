"""Streamlit demo scaffold.

Run: streamlit run demo/app.py
TODO [Team demo + phân tích hình học và cấu trúc cảnh]: upload image, pick pipeline + checkpoint, draw boxes.
"""
def main() -> None:
    try:
        import streamlit as st
    except ImportError:
        print("Install demo extras: pip install -e \".[demo]\"")
        return
    st.title("VN traffic-sign detection")
    st.caption("Scaffold — inference is not wired yet.")
    st.selectbox("Pipeline", ["faster_rcnn", "yolo26m"])
    st.file_uploader("Image", type=["jpg", "jpeg", "png"])

if __name__ == "__main__":
    main()
