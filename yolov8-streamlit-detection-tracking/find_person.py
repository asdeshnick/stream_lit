def find_person(img,db_path):
  img_copy = img.copy()
  dfs = DeepFace.find(
    img_path = img,
    db_path = db_path,
    enforce_detection=False,
    model_name = models[model_ind],
    detector_backend = backends[detect_ind],
    distance_metric = metrics[2]
    )
  sp =[]
  for i in dfs:
    try:
      display(i)
      dist_min = i["distance"].min()
      list_1= i[i["distance"] == dist_min][["identity","source_x","source_y","source_w","source_h"]].iloc[0].tolist()
      list_1[0] = extract_name(list_1[0])
      print(list_1)
      text = list_1[0]
      x,y,w,h = list_1[1:]

      cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.putText(img_copy, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
      sp.append(tuple(list_1))
    except:
      pass

  return list(set(sp)),img_copy

models = [
  "VGG-Face",  # вот эта 0!
  "Facenet",  # вот эта 1
  "Facenet512",#2
  "OpenFace", #3
  "DeepFace",  # вот эта 4
  "DeepID", # 5
  "ArcFace", #6
  "Dlib", #7
  "SFace", #8
]
model_ind = 6

backends = [
  'opencv',   # 0
  'ssd',  #1
  'dlib', #2
  'mtcnn',  #3
  'retinaface',  #4
  'mediapipe',  #5
  'yolov8',   # 6
  'yunet',  #7
  'fastmtcnn',  #8
]
detect_ind = 6

metrics = ["cosine", "euclidean", "euclidean_l2"]