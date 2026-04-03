if uploaded_files:
    new_meta = []
    new_bytes = {}

    for f in uploaded_files:
        raw = f.getvalue()
        file_hash = hashlib.md5(raw).hexdigest()[:10]
        uid = f"img_{file_hash}"

        new_bytes[uid] = raw
        new_meta.append(
            asdict(
                ImageItem(
                    id=uid,
                    original_file_name=f.name,
                    display_name="Processing..."
                )
            )
        )

    old_ids = {m["id"] for m in st.session_state["images_meta"]}
    new_ids = {m["id"] for m in new_meta}

    if old_ids != new_ids:
        st.session_state["images_bytes"] = new_bytes
        st.session_state["images_meta"] = new_meta
        st.session_state["generated_collage"] = None

        keys_to_delete = [
            k for k in list(st.session_state.keys())
            if k.startswith("dn_") or k.startswith("inp_")
        ]
        for k in keys_to_delete:
            del st.session_state[k]

        for m in st.session_state["images_meta"]:
            res = classify_image(st.session_state["images_bytes"][m["id"]])
            label = res.get("name", "Unknown Asset").strip()
            if not label:
                label = "Unknown Asset"

            st.session_state[f"dn_{m['id']}"] = label
            m["display_name"] = label

        st.rerun()
