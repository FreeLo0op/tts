import onnx
from onnx import shape_inference

# # 加载你的 ONNX 模型
# model_path = 'your_model.onnx'
# model = onnx.load(model_path)

# # 进行形状推断
# inferred_model = shape_inference.infer_shapes(model)

# # 保存推断后的模型
# inferred_model_path = 'your_model_with_shapes.onnx'
# onnx.save(inferred_model, inferred_model_path)


# /mnt/cfs/SPEECH/common/released_models/tal_vits/onnx_rhy/onnx_models/xiaosi_rhy.onnx


# /mnt/cfs/SPEECH/common/released_models/tal_vits/onnx_rhy/onnx_models/xiaosi_fear_sad_rhy.onnx

# /mnt/cfs/SPEECH/common/released_models/tal_vits/onnx_rhy/onnx_models/xiaosi_angry_rhy.onnx


# /mnt/cfs/SPEECH/common/released_models/tal_vits/onnx_rhy/onnx_models/xiaosi_rhy_eng.onnx

model_list =[
    "/mnt/cfs/SPEECH/common/released_models/tal_vits/onnx_rhy/onnx_models/xiaosi_rhy.onnx",
    "/mnt/cfs/SPEECH/common/released_models/tal_vits/onnx_rhy/onnx_models/xiaosi_fear_sad_rhy.onnx",
    "/mnt/cfs/SPEECH/common/released_models/tal_vits/onnx_rhy/onnx_models/xiaosi_angry_rhy.onnx",
    "/mnt/cfs/SPEECH/common/released_models/tal_vits/onnx_rhy/onnx_models/xiaosi_rhy_eng.onnx"
]

def add_shape():
    for model_path in model_list:
        print(f"==>> model_path: {model_path}")
        model = onnx.load(model_path)
        inferred_model = shape_inference.infer_shapes(model)
        inferred_model_path = model_path.replace(".onnx", "_with_shapes.onnx").replace("/mnt/cfs/SPEECH/common/released_models/tal_vits/onnx_rhy/onnx_models","/mnt/cfs/NLP/liujun/model/xiaosi_tts")
        onnx.save(inferred_model, inferred_model_path)
        
if __name__ == "__main__":
    add_shape()