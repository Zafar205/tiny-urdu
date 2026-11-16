from huggingface_hub import HfApi
api=HfApi()
info=api.dataset_info("kaab4321/UrduTTS",files_metadata=True)
tot=0
for s in info.siblings:
    print(s.rfilename,s.size); tot+=s.size or 0
print("total GB",tot/1e9)
print("card splits:",getattr(info.card_data,'dataset_info',None))
