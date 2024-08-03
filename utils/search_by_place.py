def search_frames_with_any_place(query_place, data, frame_ids, json_dict):
    matching_frame_indexes = []
    not_matching_frame_indexes = []
    for frame_id in frame_ids:
        frame_place = data[json_dict[str(frame_id)].split("/")[-1][:-4]]['place'].lower()
        # Kiểm tra nếu query_place có xuất hiện trong frame_place
        print(query_place, frame_place)
        if query_place.lower() in frame_place:
            matching_frame_indexes.append(frame_id)
        else :
            not_matching_frame_indexes.append(frame_id)
    id_list = matching_frame_indexes + not_matching_frame_indexes
    return id_list
