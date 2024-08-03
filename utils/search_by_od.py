def search_frames_with_all_objects(query_objects, data, frame_ids, json_dict):
    matching_frame_indexes = []
    not_matching_frame_indexes = []
    for frame_id in frame_ids:
        frame_objects = [frame_object.lower() for frame_object in data[json_dict[str(frame_id)].split("/")[-1][:-4]]['objects']]
        
        # Kiểm tra nếu tất cả các đối tượng trong query_objects có trong frame_objects
        if all(obj.lower() in frame_objects for obj in query_objects):
            matching_frame_indexes.append(frame_id)
        else:
            not_matching_frame_indexes.append(frame_id)
    id_list = matching_frame_indexes + not_matching_frame_indexes
    return id_list