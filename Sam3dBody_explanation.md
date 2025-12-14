# SAM 3D Body: Deep Dive into Model Architecture and Inference

## Overview

SAM 3D Body (3DB) is a state-of-the-art model from Meta AI for **single-image full-body 3D human mesh recovery (HMR)**. It estimates full body pose including body, feet, and hands from a single RGB image. The model is "promptable" - similar to SAM (Segment Anything Model), it can accept auxiliary prompts like 2D keypoints and segmentation masks to guide and refine predictions.

The model outputs a parametric human mesh based on the **Momentum Human Rig (MHR)**, which decouples skeletal structure from surface shape for improved accuracy.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SAM 3D Body Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input Image ──► Human Detection ──► Crop & Transform ──► Backbone     │
│                  (VitDet/YOLO)       (Affine Transform)   (DINOv3/ViT) │
│                                                                         │
│                            ▼                                            │
│                    Image Embeddings                                     │
│                            │                                            │
│            ┌───────────────┼───────────────┐                           │
│            ▼               ▼               ▼                           │
│     [Prompt Encoder]  [Ray Cond.]   [Mask Embed]                       │
│            │               │               │                           │
│            └───────────────┼───────────────┘                           │
│                            ▼                                            │
│                   Promptable Decoder                                    │
│                   (Transformer Layers)                                  │
│                            │                                            │
│            ┌───────────────┼───────────────┐                           │
│            ▼               ▼               ▼                           │
│       [MHR Head]    [Camera Head]   [Hand Decoder]                     │
│            │               │               │                           │
│            └───────────────┼───────────────┘                           │
│                            ▼                                            │
│                    ┌──────────────┐                                     │
│                    │   MHR Model  │ (Momentum Human Rig)               │
│                    │  Forward     │                                     │
│                    └──────────────┘                                     │
│                            ▼                                            │
│              3D Mesh Vertices + Keypoints                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Backbone: DINOv3/ViT Feature Extractor

**File:** `sam_3d_body/models/backbones/dinov3.py`

The backbone is a Vision Transformer (ViT) that encodes the input image into dense feature embeddings.

```python
class Dinov3Backbone(nn.Module):
    def __init__(self, name="dinov2_vitb14", ...):
        self.encoder = torch.hub.load("facebookresearch/dinov3", self.name, ...)
        self.patch_size = self.encoder.patch_size  # Typically 14
        self.embed_dim = self.encoder.embed_dim    # e.g., 1280 for ViT-H

    def forward(self, x):
        # Input: [B, 3, H, W] RGB image
        # Output: [B, C, H/patch_size, W/patch_size] feature map
        return self.encoder.get_intermediate_layers(x, reshape=True)[-1]
```

**Key details:**
- Uses DINOv3 (or ViT-H) pretrained on self-supervised tasks
- Produces spatially-organized feature maps (not just CLS token)
- Supports different model sizes: ViT-B (86M), ViT-L (307M), ViT-H+ (840M)

### 2. Input Transforms

**File:** `sam_3d_body/data/transforms/common.py`

The transform pipeline prepares human crops for the model:

```python
transform = Compose([
    GetBBoxCenterScale(padding=1.25),  # Convert bbox to center + scale
    TopdownAffine(input_size=(512, 384)),  # Affine warp to fixed size
    VisionTransformWrapper(ToTensor()),  # Convert to tensor [0, 1]
])
```

**Transform stages:**
1. **GetBBoxCenterScale:** Converts `[x1, y1, x2, y2]` bbox to center point + scale with padding
2. **TopdownAffine:** Applies affine transformation to crop the person region to a fixed aspect ratio (default 0.75 = 3:4)
3. **ToTensor:** Normalizes pixel values to [0, 1]

The affine transformation matrix is saved for later re-projection of 2D keypoints back to the original image space.

### 3. Batch Preparation

**File:** `sam_3d_body/data/utils/prepare_batch.py`

```python
def prepare_batch(img, transform, boxes, masks=None, masks_score=None, cam_int=None):
    # For each detected person bbox:
    for idx in range(boxes.shape[0]):
        data_info = dict(img=img, bbox=boxes[idx], bbox_format="xyxy")
        data_info["mask"] = masks[idx] if masks else zeros
        data_list.append(transform(data_info))

    batch = default_collate(data_list)

    # Default camera intrinsics if not provided:
    # focal_length = sqrt(height^2 + width^2)  (diagonal approximation)
    # principal_point = (width/2, height/2)
    if cam_int is None:
        batch["cam_int"] = [[focal, 0, cx], [0, focal, cy], [0, 0, 1]]

    return batch
```

**Batch contents:**
- `img`: Cropped person image tensor `[B, N_persons, 3, H, W]`
- `bbox_center`, `bbox_scale`: Bounding box parameters
- `affine_trans`: `[2, 3]` affine matrix for coordinate mapping
- `cam_int`: `[3, 3]` camera intrinsic matrix
- `mask`: Optional segmentation mask
- `mask_score`: Confidence score for mask

### 4. Prompt Encoder

**File:** `sam_3d_body/models/decoders/prompt_encoder.py`

Encodes user-provided prompts (keypoints, masks) into embeddings:

```python
class PromptEncoder(nn.Module):
    def __init__(self, embed_dim, num_body_joints=70):
        # Positional encoding using random spatial frequencies
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Per-joint learnable embeddings
        self.point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(num_body_joints)]
        self.not_a_point_embed = nn.Embedding(1, embed_dim)  # label=-1
        self.invalid_point_embed = nn.Embedding(1, embed_dim)  # label=-2

        # Optional mask downscaling CNN
        self.mask_downscaling = nn.Sequential(
            Conv2d(1, 4, kernel_size=4, stride=4), LayerNorm2d, GELU,
            Conv2d(4, 16, kernel_size=4, stride=4), LayerNorm2d, GELU,
            Conv2d(16, embed_dim, kernel_size=1)
        )

    def forward(self, keypoints):
        # keypoints: [B, N, 3] where [:, :, :2] = (x, y), [:, :, 2] = label
        # label: -2=invalid, -1=negative point, 0-69=joint index
        coords = keypoints[:, :, :2]  # Normalized to [0, 1]
        labels = keypoints[:, :, -1]

        # Fourier positional encoding
        point_embedding = self.pe_layer._pe_encoding(coords)  # sin/cos encoding

        # Add joint-specific embeddings based on label
        for i in range(num_body_joints):
            point_embedding[labels == i] += self.point_embeddings[i].weight

        return point_embedding
```

**Positional encoding:**
```python
class PositionEmbeddingRandom(nn.Module):
    def _pe_encoding(self, coords):
        # coords normalized to [0, 1], convert to [-1, 1]
        coords = 2 * coords - 1
        # Project through random Gaussian matrix
        coords = coords @ self.positional_encoding_gaussian_matrix
        # Fourier features
        return torch.cat([sin(2π * coords), cos(2π * coords)], dim=-1)
```

### 5. Promptable Transformer Decoder

**File:** `sam_3d_body/models/decoders/promptable_decoder.py`

A cross-attention decoder that processes token embeddings against image features:

```python
class PromptableDecoder(nn.Module):
    def __init__(self, dims=1024, depth=6, num_heads=8):
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                token_dims=dims,
                context_dims=context_dims,  # From backbone
                num_heads=num_heads,
                head_dims=64,
                mlp_dims=1024,
            ) for _ in range(depth)
        ])

    def forward(self, token_embedding, image_embedding, ...):
        # token_embedding: [B, N_tokens, C]
        #   - Pose token (learnable initialization)
        #   - Previous estimate token (for iterative refinement)
        #   - Prompt embeddings (keypoint prompts)
        #   - Keypoint query tokens (70 joints)
        #   - Optional 3D keypoint tokens

        # image_embedding: [B, C, H, W] flattened to [B, H*W, C]

        for layer in self.layers:
            # Cross-attention: tokens attend to image features
            # Self-attention: tokens attend to each other
            token_embedding, image_embedding = layer(
                token_embedding, image_embedding, ...)

            # Intermediate prediction for iterative refinement
            if self.do_interm_preds:
                pose_output = token_to_pose_output_fn(token_embedding)

        return self.norm_final(token_embedding), pose_outputs
```

**Token structure in decoder:**
1. **Pose token** (1): Aggregates information into pose parameters
2. **Previous estimate token** (1): Previous iteration's prediction
3. **Prompt token** (1): Keypoint prompt embedding
4. **Hand detection tokens** (2): Optional, for hand bbox prediction
5. **Keypoint tokens** (70): One per body joint for 2D regression
6. **3D Keypoint tokens** (70): One per joint for 3D regression

### 6. MHR Head (Pose Regression)

**File:** `sam_3d_body/models/heads/mhr_head.py`

Converts the pose token into MHR (Momentum Human Rig) parameters:

```python
class MHRHead(nn.Module):
    def __init__(self, input_dim=1024, mhr_model_path=""):
        # Output dimensions
        self.body_cont_dim = 260  # Body pose in continuous representation
        self.num_shape_comps = 45  # Shape PCA components
        self.num_scale_comps = 28  # Scale parameters
        self.num_hand_comps = 54   # Per-hand pose (x2 = 108 total)
        self.num_face_comps = 72   # Facial expression

        # Total output: 6 + 260 + 45 + 28 + 108 + 72 = 519 parameters
        self.npose = 6 + self.body_cont_dim + self.num_shape_comps + ...

        # MLP projection from decoder dim to pose params
        self.proj = FFN(input_dim, input_dim//8, self.npose, num_fcs=1)

        # Load MHR parametric body model
        self.mhr = torch.jit.load(mhr_model_path)  # TorchScript model

    def forward(self, x, init_estimate=None):
        # x: [B, C] pose token
        pred = self.proj(x)  # [B, npose]
        if init_estimate is not None:
            pred = pred + init_estimate  # Residual prediction

        # Parse predictions
        global_rot_6d = pred[:, :6]  # 6D rotation representation
        body_pose_cont = pred[:, 6:266]  # Continuous body pose
        shape = pred[:, 266:311]  # Shape parameters
        scale = pred[:, 311:339]  # Scale parameters
        hand = pred[:, 339:447]  # Hand pose (left + right)
        face = pred[:, 447:519]  # Face expression

        # Convert 6D to rotation matrix
        global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)
        global_rot_euler = roma.rotmat_to_euler("ZYX", global_rot_rotmat)

        # Convert continuous body pose to euler angles
        body_pose_euler = compact_cont_to_model_params_body(body_pose_cont)

        # Run MHR forward pass
        output = self.mhr_forward(
            global_trans=zeros, global_rot=global_rot_euler,
            body_pose_params=body_pose_euler, hand_pose_params=hand,
            scale_params=scale, shape_params=shape, expr_params=face,
            return_keypoints=True, return_vertices=True
        )

        return output  # vertices, 3D keypoints, joint coords, etc.
```

**MHR Model output:**
```python
def mhr_forward(self, global_trans, global_rot, body_pose, hand_pose, scale, shape, expr):
    # Compute scales from PCA
    scales = self.scale_mean + scale @ self.scale_comps  # [B, 68]

    # Concatenate all pose parameters
    model_params = cat([global_trans*10, global_rot, body_pose, hand_pose, scales])

    # Run MHR (parametric body model)
    skinned_verts, skel_state = self.mhr(shape, model_params, expr)
    # skinned_verts: [B, ~18439, 3] mesh vertices
    # skel_state: [B, 127, 8] joint coords + quaternions + scale

    joint_coords, joint_quats, _ = split(skel_state, [3, 4, 1], dim=2)

    # Get 70 keypoints from vertices + joints
    keypoints = self.keypoint_mapping @ cat([skinned_verts, joint_coords])

    # Fix camera coordinate system (Y-up to Z-forward)
    verts[..., [1, 2]] *= -1
    keypoints[..., [1, 2]] *= -1

    return verts, keypoints, joint_coords, joint_rotations
```

### 7. Camera Head (Translation Prediction)

**File:** `sam_3d_body/models/heads/camera_head.py`

Predicts camera translation for perspective projection:

```python
class PerspectiveHead(nn.Module):
    def __init__(self, input_dim, img_size):
        self.ncam = 3  # (s, tx, ty) - scale and 2D translation
        self.proj = FFN(input_dim, input_dim//8, self.ncam, num_fcs=1)

    def forward(self, x, init_estimate=None):
        pred_cam = self.proj(x)  # [B, 3]
        return pred_cam + init_estimate if init_estimate else pred_cam

    def perspective_projection(self, points_3d, pred_cam, bbox_center,
                                bbox_size, img_size, cam_int):
        # pred_cam: (s, tx, ty) - scale and 2D offset
        s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]

        # Compute depth from scale: tz = 2 * focal_length / (bbox_size * s)
        focal_length = cam_int[:, 0, 0]
        tz = 2 * focal_length / (bbox_size * s)

        # Compute 3D translation from bbox center offset
        cx = 2 * (bbox_center[:, 0] - img_size[:, 0]/2) / (bbox_size * s)
        cy = 2 * (bbox_center[:, 1] - img_size[:, 1]/2) / (bbox_size * s)
        pred_cam_t = stack([tx + cx, ty + cy, tz])  # [B, 3]

        # Translate 3D points to camera frame
        j3d_cam = points_3d + pred_cam_t.unsqueeze(1)

        # Project to 2D: j2d = K @ (j3d / j3d_z)
        j2d = perspective_projection(j3d_cam, cam_int)

        return {
            "pred_keypoints_2d": j2d,
            "pred_cam_t": pred_cam_t,
            "focal_length": focal_length
        }
```

**The CLIFF-style camera formulation:**
- Instead of predicting absolute 3D translation, predict relative to bbox
- Scale `s` controls perceived depth (larger s = closer to camera)
- `tx, ty` are 2D offsets in normalized bbox coordinates

---

## Inference Pipeline

### Entry Point: `SAM3DBodyEstimator.process_one_image()`

**File:** `sam_3d_body/sam_3d_body_estimator.py`

```python
class SAM3DBodyEstimator:
    @torch.no_grad()
    def process_one_image(self, img, bboxes=None, masks=None, cam_int=None, ...):
        # 1. Load image
        if type(img) == str:
            img = cv2.imread(img)  # BGR format

        # 2. Human detection (if no bboxes provided)
        if bboxes is None and self.detector:
            boxes = self.detector.run_human_detection(img, bbox_thr=0.8)
        else:
            boxes = bboxes or [[0, 0, width, height]]  # Full image

        # 3. Optional segmentation masks
        if masks is None and use_mask and self.sam:
            masks, masks_score = self.sam.run_sam(img, boxes)

        # 4. Prepare batch
        batch = prepare_batch(img, self.transform, boxes, masks)
        batch = recursive_to(batch, self.device)

        # 5. Initialize model internal state
        self.model._initialize_batch(batch)

        # 6. Camera intrinsics (from FOV estimator or default)
        if self.fov_estimator:
            cam_int = self.fov_estimator.get_cam_intrinsics(batch["img_ori"])
        batch["cam_int"] = cam_int

        # 7. Run inference
        outputs = self.model.run_inference(
            img, batch, inference_type="full",
            transform_hand=self.transform_hand
        )

        # 8. Format outputs
        return [{
            "bbox": batch["bbox"][i],
            "focal_length": out["focal_length"][i],
            "pred_keypoints_3d": out["pred_keypoints_3d"][i],  # [70, 3]
            "pred_keypoints_2d": out["pred_keypoints_2d"][i],  # [70, 2]
            "pred_vertices": out["pred_vertices"][i],          # [18439, 3]
            "pred_cam_t": out["pred_cam_t"][i],                # [3]
            "body_pose_params": out["body_pose"][i],           # Euler angles
            "hand_pose_params": out["hand"][i],                # Hand PCA
            "shape_params": out["shape"][i],                   # Body shape
            "scale_params": out["scale"][i],                   # Body scale
        } for i in range(N_persons)]
```

### Full-Body Inference Pipeline: `run_inference()`

**File:** `sam_3d_body/models/meta_arch/sam3d_body.py`

```python
def run_inference(self, img, batch, inference_type="full", transform_hand=None):
    """
    inference_type options:
    - "body": Body decoder only (fast, less hand detail)
    - "hand": Hand decoder only (for hand crops)
    - "full": Body + hand refinement (best quality)
    """

    # Step 1: Run body decoder
    pose_output = self.forward_step(batch, decoder_type="body")

    if inference_type != "full":
        return pose_output

    # Step 2: Extract hand bounding boxes from body prediction
    left_xyxy, right_xyxy = self._get_hand_box(pose_output, batch)

    # Step 3: Run hand decoder on left hand (flipped)
    flipped_img = img[:, ::-1]
    left_xyxy_flipped = flip_x_coords(left_xyxy)
    batch_lhand = prepare_batch(flipped_img, transform_hand, left_xyxy_flipped)
    lhand_output = self.forward_step(batch_lhand, decoder_type="hand")
    # Unflip the hand pose

    # Step 4: Run hand decoder on right hand
    batch_rhand = prepare_batch(img, transform_hand, right_xyxy)
    rhand_output = self.forward_step(batch_rhand, decoder_type="hand")

    # Step 5: Validate hand predictions
    # Multiple criteria: wrist angle consistency, box size, 2D keypoint bounds
    hand_valid_mask = (
        angle_difference_valid &
        box_size_valid &
        kps_in_bounds &
        wrist_distance_valid
    )

    # Step 6: Keypoint prompting - refine body using hand wrist locations
    keypoint_prompt = create_prompt_from_wrists(
        lhand_output, rhand_output, pose_output
    )
    if keypoint_prompt.numel():
        pose_output = self.run_keypoint_prompt(batch, pose_output, keypoint_prompt)

    # Step 7: Replace hand poses from hand decoder into body output
    pose_output["mhr"]["hand"] = cat([lhand_hand_pose, rhand_hand_pose])
    pose_output["mhr"]["scale"][:, 8:10] = hand_scales

    return pose_output, batch_lhand, batch_rhand, ...
```

### Forward Pass: `forward_step()` and `forward_pose_branch()`

```python
def forward_pose_branch(self, batch):
    # 1. Preprocess images
    x = self.data_preprocess(batch["img"])  # Normalize, crop width for ViT

    # 2. Compute ray conditioning (for camera-aware features)
    ray_cond = self.get_ray_condition(batch)  # [B, N, 2, H, W]

    # 3. Extract image features
    image_embeddings = self.backbone(x)  # [B*N, C, H', W']

    # 4. Add mask embeddings if available
    if use_mask:
        mask_embeddings = self._get_mask_prompt(batch, image_embeddings)
        image_embeddings = image_embeddings + mask_embeddings

    # 5. Compute decoder condition (CLIFF-style)
    condition_info = self._get_decoder_condition(batch)  # [B*N, 3]
    # Contains: (cx - img_w/2)/f, (cy - img_h/2)/f, bbox_size/f

    # 6. Create initial keypoint prompt (dummy - all invalid)
    keypoints_prompt = zeros([B*N, 1, 3])
    keypoints_prompt[:, :, -1] = -2  # Label -2 = invalid

    # 7. Run promptable decoder
    tokens_output, pose_output = self.forward_decoder(
        image_embeddings,
        keypoints=keypoints_prompt,
        condition_info=condition_info,
        batch=batch
    )

    return {"mhr": pose_output[-1], "image_embeddings": image_embeddings}
```

### Decoder Forward: `forward_decoder()`

```python
def forward_decoder(self, image_embeddings, keypoints, condition_info, batch):
    B = image_embeddings.shape[0]

    # 1. Initialize pose token
    init_pose = self.init_pose.weight.expand(B, -1).unsqueeze(1)  # [B, 1, 519]
    init_camera = self.init_camera.weight.expand(B, -1).unsqueeze(1)  # [B, 1, 3]
    init_estimate = cat([init_pose, init_camera], dim=-1)  # [B, 1, 522]

    # 2. Add condition info
    init_input = cat([condition_info.unsqueeze(1), init_estimate], dim=-1)  # [B, 1, 525]

    # 3. Project to decoder dimension
    token_embeddings = self.init_to_token_mhr(init_input)  # [B, 1, 1024]

    # 4. Add prompt embeddings
    if keypoints is not None:
        prompt_embeddings = self.prompt_encoder(keypoints)  # [B, N_prompts, 1280]
        prompt_embeddings = self.prompt_to_token(prompt_embeddings)  # [B, N, 1024]
        token_embeddings = cat([token_embeddings, prev_embeddings, prompt_embeddings], dim=1)

    # 5. Add keypoint query tokens
    token_embeddings = cat([
        token_embeddings,
        self.keypoint_embedding.weight.expand(B, -1, -1),  # 70 2D keypoint tokens
        self.keypoint3d_embedding.weight.expand(B, -1, -1)  # 70 3D keypoint tokens
    ], dim=1)

    # 6. Add positional encoding to image features
    image_augment = self.prompt_encoder.get_dense_pe(image_embeddings.shape[-2:])
    image_embeddings = self.ray_cond_emb(image_embeddings, batch["ray_cond"])

    # 7. Run transformer decoder
    def token_to_pose_output_fn(tokens, prev_output, layer_idx):
        pose_token = tokens[:, 0]  # First token is pose token
        pose_output = self.head_pose(pose_token, init_pose)
        cam_output = self.head_camera(pose_token, init_camera)
        pose_output["pred_cam"] = cam_output
        pose_output = self.camera_project(pose_output, batch)
        return pose_output

    pose_token, pose_outputs = self.decoder(
        token_embeddings, image_embeddings,
        token_to_pose_output_fn=token_to_pose_output_fn
    )

    return pose_token, pose_outputs  # List of outputs from each layer
```

---

## Output Format

### Per-Person Output Dictionary

```python
output = {
    # Bounding box (original image coordinates)
    "bbox": np.array([x1, y1, x2, y2]),  # [4]

    # Camera parameters
    "focal_length": float,  # Estimated focal length in pixels
    "pred_cam_t": np.array([tx, ty, tz]),  # 3D camera translation [3]

    # 3D Predictions (in camera coordinates, meters)
    "pred_keypoints_3d": np.array(...),  # [70, 3] - 70 body keypoints
    "pred_vertices": np.array(...),       # [18439, 3] - Mesh vertices
    "pred_joint_coords": np.array(...),   # [127, 3] - Skeleton joint positions
    "pred_global_rots": np.array(...),    # [127, 3, 3] - Joint rotation matrices

    # 2D Predictions (original image coordinates)
    "pred_keypoints_2d": np.array(...),   # [70, 2] - Projected keypoints

    # MHR Model Parameters
    "global_rot": np.array(...),          # [3] - Global rotation (euler ZYX)
    "body_pose_params": np.array(...),    # [130] - Body joint angles (euler)
    "hand_pose_params": np.array(...),    # [108] - Hand PCA coefficients
    "shape_params": np.array(...),        # [45] - Body shape PCA
    "scale_params": np.array(...),        # [28] - Body proportions
    "expr_params": np.array(...),         # [72] - Facial expression

    # Raw continuous predictions
    "pred_pose_raw": np.array(...),       # [266] - 6D rot + continuous body pose

    # Optional mask
    "mask": np.array(...) or None,        # [H, W, 1] - Segmentation mask

    # Hand bounding boxes (if full inference)
    "lhand_bbox": np.array([x1, y1, x2, y2]),
    "rhand_bbox": np.array([x1, y1, x2, y2]),
}
```

### 70 Keypoint Definitions

**File:** `sam_3d_body/metadata/mhr70.py`

```
Index 0-4:   Face (nose, eyes, ears)
Index 5-10:  Upper body (shoulders, elbows, hips)
Index 11-14: Lower body (knees, ankles)
Index 15-20: Feet (big toe, small toe, heel x2)
Index 21-41: Right hand (thumb, index, middle, ring, pinky + wrist)
Index 42-62: Left hand (same structure)
Index 63-69: Extra (olecranon, cubital fossa, acromion, neck)
```

---

## Key Design Decisions

### 1. 6D Rotation Representation
The model uses 6D rotation representation (first two columns of rotation matrix) instead of axis-angle or quaternions because it's continuous and avoids gimbal lock issues during training.

```python
def rot6d_to_rotmat(x):
    # x: [B, 6] -> [B, 3, 3]
    a1, a2 = x[:, :3], x[:, 3:6]
    b1 = normalize(a1)
    b2 = normalize(a2 - (b1 @ a2) * b1)  # Gram-Schmidt
    b3 = cross(b1, b2)
    return stack([b1, b2, b3], dim=-1)
```

### 2. CLIFF-style Camera Prediction
Instead of predicting absolute 3D translation, the model predicts scale and offset relative to the bounding box. This makes predictions more robust to different image resolutions and focal lengths.

### 3. Two-Stage Hand Refinement
For full-body inference:
1. Body decoder predicts coarse hand locations
2. Hand decoder runs on cropped hand regions
3. Results are validated and merged back

This allows the hand decoder to focus on detailed finger articulation while the body decoder handles overall pose.

### 4. Iterative Refinement with Prompts
The model supports keypoint prompting for iterative refinement:
1. Initial prediction without prompts
2. User provides corrective keypoints
3. Model refines prediction conditioned on prompts

---

## Dependencies

### Core Libraries
- **PyTorch 2.0+**: Neural network framework
- **torchvision**: Image transforms
- **roma**: Rotation matrix operations (`pip install roma`)
- **OpenCV**: Image I/O and affine transforms

### Optional Components
- **VitDet** (detectron2): Human detection
- **SAM2**: Instance segmentation for masks
- **MoGe2**: Field-of-view estimation
- **MHR** (Momentum Human Rig): Parametric body model (loaded as TorchScript)

### Model Sizes
| Variant | Backbone | Total Params | Input Size |
|---------|----------|--------------|------------|
| ViT-H | ViT-Huge | 631M | 512x384 |
| DINOv3-H+ | DINOv3-Huge | 840M | 512x384 |

---

## Usage Example

```python
import cv2
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

# Load model
model, cfg = load_sam_3d_body(
    checkpoint_path="checkpoints/model.ckpt",
    mhr_path="checkpoints/mhr_model.pt",
    device="cuda"
)

# Create estimator
estimator = SAM3DBodyEstimator(
    sam_3d_body_model=model,
    model_cfg=cfg,
    human_detector=None,  # Optional VitDet
    fov_estimator=None,   # Optional MoGe2
)

# Process image
img = cv2.imread("person.jpg")
outputs = estimator.process_one_image(
    img,
    bboxes=np.array([[100, 50, 400, 600]]),  # Optional manual bbox
    use_mask=False,
    inference_type="full"  # or "body" for faster
)

# Access results
for person in outputs:
    vertices = person["pred_vertices"]  # [18439, 3] mesh
    keypoints_3d = person["pred_keypoints_3d"]  # [70, 3]
    keypoints_2d = person["pred_keypoints_2d"]  # [70, 2]
```

---

## Summary

SAM 3D Body is a sophisticated human mesh recovery system that:

1. **Encodes images** using a pretrained ViT/DINOv3 backbone
2. **Accepts optional prompts** (keypoints, masks) through a prompt encoder
3. **Decodes pose parameters** via a cross-attention transformer
4. **Regresses MHR parameters** for body pose, shape, and hand articulation
5. **Projects to 2D** using a learned camera model
6. **Optionally refines hands** using a dedicated hand decoder

The architecture is designed for robust generalization across diverse poses, viewpoints, and occlusions while maintaining high-fidelity hand reconstruction.

---

## MHR (Momentum Human Rig): Complete Primer

### What is MHR?

**MHR (Momentum Human Rig)** is Meta's parametric human body model that powers SAM 3D Body. Unlike SMPL/SMPL-X which use blend shapes and linear blend skinning with a single skeleton, MHR **decouples skeletal structure from surface shape**:

1. **Skeleton**: 127 joints with hierarchical rotations
2. **Shape**: 45-dimensional PCA space for body shape variation (thin/heavy, tall/short)
3. **Scale**: 68-dimensional per-bone scaling (arm length, torso width, etc.)
4. **Mesh**: 18,439 vertices for the body surface

This decoupling allows MHR to represent body proportions more accurately than blend-shape-only models.

---

### Units and Coordinate Systems

#### Units: **Meters**

All 3D outputs from SAM 3D Body are in **meters**:
- `pred_vertices`: [18439, 3] mesh vertex positions in meters
- `pred_keypoints_3d`: [70, 3] keypoint positions in meters
- `pred_joint_coords`: [127, 3] skeleton joint positions in meters
- `pred_cam_t`: [3] camera translation in meters

**Note:** Internally, MHR works in centimeters and multiplies by 10 for stability. The outputs are converted back:
```python
# Inside mhr_forward():
skinned_verts = skinned_verts / 100  # cm → meters
joint_coords = joint_coords / 100    # cm → meters
```

#### Coordinate System: **Camera Frame (OpenCV Convention)**

The model outputs are in **camera coordinates**:

```
        Y (up in image, but NEGATED in 3D)
        ↑
        │
        │
        │
        └──────→ X (right in image)
       /
      /
     ↓
    Z (depth - into the scene, NEGATED)
```

**CRITICAL:** SAM 3D Body applies a coordinate flip for rendering compatibility:
```python
verts[..., [1, 2]] *= -1     # Flip Y and Z
keypoints[..., [1, 2]] *= -1  # Flip Y and Z
```

This means:
- **+X** = right in the image
- **+Y** = down in 3D (up in image after negation)
- **+Z** = behind the camera (into scene after negation)

For most 3D engines (Unity, Unreal, Blender), you'll need to convert to their coordinate system.

---

### The 70 Keypoints (pred_keypoints_3d)

These are semantic body landmarks derived from the mesh vertices and skeleton joints:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MHR 70 KEYPOINT MAP                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  HEAD (0-4):                                                            │
│    0: nose                                                              │
│    1: left_eye          2: right_eye                                    │
│    3: left_ear          4: right_ear                                    │
│                                                                         │
│  UPPER BODY (5-10):                                                     │
│    5: left_shoulder     6: right_shoulder                               │
│    7: left_elbow        8: right_elbow                                  │
│    9: left_hip         10: right_hip                                    │
│                                                                         │
│  LOWER BODY (11-14):                                                    │
│   11: left_knee        12: right_knee                                   │
│   13: left_ankle       14: right_ankle                                  │
│                                                                         │
│  FEET (15-20):                                                          │
│   15: left_big_toe     18: right_big_toe                                │
│   16: left_small_toe   19: right_small_toe                              │
│   17: left_heel        20: right_heel                                   │
│                                                                         │
│  RIGHT HAND (21-41):                                                    │
│   21-24: right_thumb (tip → third_joint)                                │
│   25-28: right_index (tip → third_joint)                                │
│   29-32: right_middle (tip → third_joint)                               │
│   33-36: right_ring (tip → third_joint)                                 │
│   37-40: right_pinky (tip → third_joint)                                │
│   41: right_wrist                                                       │
│                                                                         │
│  LEFT HAND (42-62):                                                     │
│   42-45: left_thumb (tip → third_joint)                                 │
│   46-49: left_index (tip → third_joint)                                 │
│   50-53: left_middle (tip → third_joint)                                │
│   54-57: left_ring (tip → third_joint)                                  │
│   58-61: left_pinky (tip → third_joint)                                 │
│   62: left_wrist                                                        │
│                                                                         │
│  EXTRA LANDMARKS (63-69):                                               │
│   63: left_olecranon (back of elbow)                                    │
│   64: right_olecranon                                                   │
│   65: left_cubital_fossa (inner elbow)                                  │
│   66: right_cubital_fossa                                               │
│   67: left_acromion (shoulder tip)                                      │
│   68: right_acromion                                                    │
│   69: neck                                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Python reference:**
```python
KEYPOINT_NAMES = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_hip", 10: "right_hip", 11: "left_knee", 12: "right_knee",
    13: "left_ankle", 14: "right_ankle",
    15: "left_big_toe", 16: "left_small_toe", 17: "left_heel",
    18: "right_big_toe", 19: "right_small_toe", 20: "right_heel",
    # Right hand: 21-41
    21: "right_thumb_tip", 22: "right_thumb_first_joint",
    23: "right_thumb_second_joint", 24: "right_thumb_third_joint",
    25: "right_index_tip", ..., 41: "right_wrist",
    # Left hand: 42-62 (same structure)
    42: "left_thumb_tip", ..., 62: "left_wrist",
    # Extra
    63: "left_olecranon", 64: "right_olecranon",
    65: "left_cubital_fossa", 66: "right_cubital_fossa",
    67: "left_acromion", 68: "right_acromion", 69: "neck"
}
```

---

### The 127 Skeleton Joints (pred_joint_coords)

The MHR skeleton has **127 joints** organized hierarchically. Key joint indices:

```python
# Core body joints (approximate - exact indices from MHR rig)
SKELETON_JOINTS = {
    # Spine chain
    0: "root/pelvis",
    1: "spine_01",
    2: "spine_02",
    3: "spine_03",
    4: "neck",
    5: "head",

    # Left leg
    9: "left_hip",       # (same as keypoint 9)
    11: "left_knee",     # (same as keypoint 11)
    13: "left_ankle",    # (same as keypoint 13)

    # Right leg
    10: "right_hip",     # (same as keypoint 10)
    12: "right_knee",    # (same as keypoint 12)
    14: "right_ankle",   # (same as keypoint 14)

    # Left arm
    5: "left_shoulder",
    7: "left_elbow",
    76: "left_lowarm",   # Lower arm
    77: "left_wrist_twist",
    78: "left_wrist",

    # Right arm
    6: "right_shoulder",
    8: "right_elbow",
    40: "right_lowarm",  # Lower arm
    41: "right_wrist_twist",
    42: "right_wrist",

    # Left hand (16 joints per hand)
    # 79-94: left hand finger joints

    # Right hand
    # 43-58: right hand finger joints

    # Feet
    # Additional toe joints...
}
```

**Important joint indices used in code:**
```python
lowarm_joint_idxs = [76, 40]         # left_lowarm, right_lowarm
wrist_twist_joint_idxs = [77, 41]    # left_wrist_twist, right_wrist_twist
left_wrist_idx = 78
right_wrist_idx = 42
```

---

### Understanding the Output Values

#### 1. `pred_vertices` - Mesh Vertices
```python
vertices = output["pred_vertices"]  # [18439, 3]
# Each row is (x, y, z) in meters, camera frame
# These define the body surface mesh
```

#### 2. `pred_cam_t` - Camera Translation
```python
cam_t = output["pred_cam_t"]  # [3] = [tx, ty, tz]
# Translation from camera origin to body root, in meters
# tz is depth (distance from camera), typically 2-10 meters
```

The body is predicted in a **root-relative frame**, then translated by `pred_cam_t` to get camera-frame positions:
```python
# To get world positions from model output:
vertices_in_camera = pred_vertices + pred_cam_t  # Add translation
```

#### 3. `global_rot` - Global Body Orientation
```python
global_rot = output["global_rot"]  # [3] Euler angles (ZYX order)
# Rotation of the entire body relative to camera
# In radians, order: [yaw, pitch, roll]
```

**Why ZYX order?** This is a common convention for computer vision and robotics:
- **First rotate around Z (yaw)**: Turn left/right (like shaking head "no")
- **Then rotate around Y (pitch)**: Nod up/down (like saying "yes")
- **Finally rotate around X (roll)**: Tilt head sideways (like touching ear to shoulder)

**Note**: The `global_rot` is computed from `roma.rotmat_to_euler("ZYX", rotation_matrix)`, which means the rotation matrix is decomposed into ZYX Euler angles.

The global rotation defines how the person is oriented relative to the camera:
- **Facing camera straight**: global_rot ≈ [0, 0, 0]
- **Turned 90° left**: global_rot ≈ [π/2, 0, 0] (positive yaw = left turn)
- **Turned 90° right**: global_rot ≈ [-π/2, 0, 0] (negative yaw = right turn)
- **Looking up 30°**: global_rot ≈ [0, -π/6, 0] (negative pitch = look up)
- **Looking down 30°**: global_rot ≈ [0, π/6, 0] (positive pitch = look down)
- **Head tilted right**: global_rot ≈ [0, 0, π/6] (positive roll = tilt right)

**Camera Coordinate System Axes:**
```
         +Y (up in image, NEGATED in 3D output)
          ↑
          |
    +X ←─+──→ +X (right in image)
         |
         ↓
         +Z (depth, NEGATED - into scene)
```

For rotations:
- **X-axis**: Roll - tilt left/right
- **Y-axis**: Pitch - look up/down
- **Z-axis**: Yaw - turn left/right

---

### Concrete Example Values

Here are typical values you'll see from SAM 3D Body, with interpretations:

```python
# Example output for a person 3m from camera
output = {
    "pred_cam_t": [-0.15, 0.08, 3.02],           # [tx, ty, tz] in meters
    "global_rot": [0.12, -0.05, 0.02],           # [yaw, pitch, roll] in radians
    "focal_length": 850.0,                        # pixels

    # Key points in camera frame (meters)
    "pred_keypoints_3d": [
        # Head
        [0.0, 0.0, 0.0],         # 0: nose (at origin before cam_t applied)
        [-0.03, 0.06, -0.05],    # 1: left_eye
        [0.03, 0.06, -0.05],     # 2: right_eye

        # Body
        [-0.18, 0.25, 0.02],     # 5: left_shoulder
        [0.18, 0.25, 0.02],      # 6: right_shoulder
        [0.0, -0.12, 0.03],      # 9: left_hip
        [0.0, -0.12, -0.03],     # 10: right_hip

        # Arms (extended)
        [-0.45, 0.20, 0.10],     # 7: left_elbow
        [0.48, 0.18, -0.05],     # 8: right_elbow
        [-0.65, 0.10, 0.20],     # 41: right_wrist
        [0.60, 0.05, -0.15],     # 62: left_wrist

        # Legs
        [-0.08, -0.48, 0.05],    # 11: left_knee
        [0.08, -0.48, -0.03],    # 12: right_knee
        [-0.09, -0.95, 0.08],    # 13: left_ankle
        [0.09, -0.95, -0.05],    # 14: right_ankle
    ],

    # Body pose parameters (130 values, first few shown)
    "body_pose_params": [
        0.015, -0.021, 0.008,     # Joint 0-2: root rotations (XYZ euler)
        0.032, -0.015, -0.018,    # Joint 3-5: spine rotations
        -0.05, 0.12, 0.0,        # Joint 6-8: left shoulder
        0.08, -0.11, 0.02,       # Joint 9-11: right shoulder
        1.23, 0.0, 0.0,          # Joint 12-14: left elbow (1.23 rad = 70° bend)
        -1.15, 0.0, 0.0,         # Joint 15-17: right elbow (-1.15 rad = -66° bend)
        -0.5, 0.0, 0.0,          # Joint 18-20: left knee (-0.5 rad = -28° bend)
        0.48, 0.0, 0.0,          # Joint 21-23: right knee (0.48 rad = 28° bend)
        # ... 107 more joint parameters
    ],

    # Shape parameters (PCA coefficients)
    "shape_params": [
        1.23,    # 0: taller than average
        -0.87,   # 1: thinner than average
        0.45,    # 2: slightly muscular
        # ... 42 more values (typically -3 to +3 range)
    ],

    # Scale parameters (28 values)
    "scale_params": [
        0.02,   # 0: pelvis width
        0.15,   # 1: torso height
        -0.05,  # 2: arm length
        0.12,   # 3: leg length
        # ... 24 more values
    ],

    # 2D projections (pixels) - assuming 1024x1024 image
    "pred_keypoints_2d": [
        [512, 380],          # 0: nose (center top third)
        [495, 360],          # 1: left_eye
        [529, 360],          # 2: right_eye
        [380, 500],          # 5: left_shoulder
        [644, 500],          # 6: right_shoulder
        [480, 750],          # 13: left_ankle
        [544, 750],          # 14: right_ankle
    ],
}
```

#### Interpreting These Values:

1. **Camera Translation `[-0.15, 0.08, 3.02]`**:
   - Person is 3.02m from camera
   - Slightly to the left (-0.15m) and up (0.08m) in camera view
   - This is applied to ALL vertices: `final_vertices = vertices + cam_t`

2. **Global Rotation `[0.12, -0.05, 0.02]`**:
   - `0.12 rad = 6.9°`: Turned slightly to left
   - `-0.05 rad = -2.9°`: Looking slightly up
   - `0.02 rad = 1.1°`: Head tilted slightly right

3. **Body Pose - Elbow Bending**:
   - Left elbow: `1.23 rad = 70.5°` bend
   - Right elbow: `-1.15 rad = -65.9°` bend
   - Positive means flexing (bringing hand toward shoulder)
   - Negative means extending

4. **Keypoint Positions**:
   - Before `pred_cam_t`: Keypoints are relative to body center at origin
   - After adding `pred_cam_t`: All positions shift to camera frame
   - Example: Right hand final position = `[-0.65, 0.10, 0.20] + [-0.15, 0.08, 3.02] = [-0.80, 0.18, 3.22]`

5. **2D Projections**:
   - Computed using camera intrinsics and perspective projection
   - `pixel_x = fx * (X/Z) + cx`
   - `pixel_y = fy * (Y/Z) + cy`
   - Where `[cx, cy]` is image center

#### More Extreme Examples:

```python
# Person far away (8m)
output_far = {
    "pred_cam_t": [0.02, -0.01, 8.15],
    # Smaller in image, similar pose
}

# Person close (1.5m)
output_close = {
    "pred_cam_t": [-0.05, 0.12, 1.52],
    # Larger in image
}

# Person lying on ground
output_lying = {
    "global_rot": [0, 1.57, 0],  # 90° pitch (lying down)
    "pred_keypoints_3d": [
        [0.0, 0.0, 0.0],         # Head on ground
        [0.0, 1.75, 0.0],        # Feet at body height
    ],
}

# Person with arms raised
output_arms_up = {
    "body_pose_params": [
        # ... shoulder joints
        0.0, 1.5, 0.0,         # Left shoulder raised 90°
        0.0, -1.5, 0.0,        # Right shoulder raised -90°
        # ...
    ],
}
```

**Remember**: All rotations follow right-hand rule with the coordinate system shown above.

#### 4. `pred_global_rots` - Per-Joint World Rotations
```python
joint_rots = output["pred_global_rots"]  # [127, 3, 3] rotation matrices
# Each joint's orientation in world/camera frame
# NOT local rotations - these are accumulated down the kinematic chain
```

#### 5. `body_pose_params` - Local Joint Angles
```python
body_pose = output["body_pose_params"]  # [130] Euler angles
# Local rotation of each joint relative to its parent
# Ordered by joint index, 3 values (XYZ euler) per joint for 3-DOF joints
# Some joints have 1-DOF (hinge) or 2-DOF
```

**Joint Rotation Axes (XYZ order):**
For each joint with 3 DOF, the rotation order is X-Y-Z:
- **X-axis**: Flexion/extension (bend/straighten)
  - Elbow/knee: Positive = flex (bend), Negative = extend
  - Shoulder: Positive = raise forward, Negative = lower
- **Y-axis**: Abduction/adduction (move away from/toward body)
  - Arm: Positive = raise sideways, Negative = lower
  - Leg: Positive = spread apart, Negative = bring together
- **Z-axis**: Internal/external rotation (twist)
  - Shoulder/hip: Positive = rotate inward, Negative = rotate outward

**Joint DOF Structure:**
```python
# Example joint configurations:
# 3-DOF joints (3 parameters):
spine_joints = [X, Y, Z]      # Can bend/rotate in all directions
shoulder_joints = [X, Y, Z]  # Forward/back, sideways, twist
hip_joints = [X, Y, Z]       # Forward/back, sideways, twist

# 2-DOF joints (2 parameters):
knee_joints = [X, Y]         # Bend/extend, slight twist
elbow_joints = [X, Y]        # Bend/extend, twist forearm

# 1-DOF joints (1 parameter):
toe_joints = [X]             # Curl/extend toes
finger_joints = [X]          # Curl/extend fingers
```

#### 6. `shape_params` - Body Shape
```python
shape = output["shape_params"]  # [45] PCA coefficients
# Controls body shape: thin/heavy, muscular, etc.
# shape[0:10] are the most significant components
```

#### 7. `scale_params` - Body Proportions
```python
scale = output["scale_params"]  # [28] PCA coefficients
# Controls limb lengths, torso proportions
# Different from shape - this affects skeleton, not just surface
```

Key scale indices:
```python
scale[8]  # Right hand scale
scale[9]  # Left hand scale
scale[18:] # Finger scales
```

#### 8. `focal_length` - Camera Focal Length
```python
focal = output["focal_length"]  # scalar, in pixels
# Estimated from image or provided
# Used for perspective projection
```

---

### Placing a Player in a 3D Scene: Complete Workflow

Let's walk through the complete process from model output to a person placed in your 3D scene.

#### Step 1: Get Model Output

```python
import cv2
import numpy as np
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

# Load the model
model, cfg = load_sam_3d_body(
    checkpoint_path="checkpoints/model.ckpt",
    mhr_path="checkpoints/mhr_model.pt"
)
estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)

# Process image
img = cv2.imread("person.jpg")
outputs = estimator.process_one_image(img)
output = outputs[0]  # First person (if multiple)
```

**What you get:**
```python
output = {
    # Mesh data (root-relative)
    "pred_vertices": np.array([[0.15, 1.2, -0.05], ...]),  # [18439, 3] in meters
    "faces": np.array([[0, 1, 2], ...]),                   # [36874, 3] triangle indices

    # Camera frame position
    "pred_cam_t": np.array([-0.15, 0.08, 3.02]),           # [tx, ty, tz] in meters

    # Body orientation
    "global_rot": np.array([0.12, -0.05, 0.02]),           # [yaw, pitch, roll] in radians

    # Keypoints (root-relative, before cam_t applied)
    "pred_keypoints_3d": np.array([[0.0, 0.0, 0.0], ...]), # [70, 3] in meters
    "pred_keypoints_2d": np.array([[512, 380], ...]),       # [70, 2] in pixels

    # Camera info
    "focal_length": 850.0,                                 # pixels
}
```

#### Step 2: Convert to Camera Frame Coordinates

The model outputs are **root-relative**. Add the camera translation to get actual camera-frame positions:

```python
def to_camera_frame(vertices, keypoints, cam_t):
    """Convert root-relative coordinates to camera frame."""
    # Apply camera translation to all points
    vertices_cam = vertices + cam_t.reshape(1, 3)
    keypoints_cam = keypoints + cam_t.reshape(1, 3)
    return vertices_cam, keypoints_cam

# Apply transformation
vertices_cam, keypoints_cam = to_camera_frame(
    output["pred_vertices"],
    output["pred_keypoints_3d"],
    output["pred_cam_t"]
)

# Now:
# - vertices_cam: mesh vertices in camera space
# - keypoints_cam: landmarks in camera space
```

#### Step 3: Handle Coordinate System Conversion

SAM 3D Body uses a flipped Y/Z coordinate system. Convert to your engine's convention:

```python
def sam3d_to_world_coords(points_cam, target_system="y-up"):
    """
    Convert from SAM3D camera coordinates to world coordinates.

    Args:
        points_cam: [N, 3] points in SAM3D camera frame
        target_system: "y-up" (Blender), "z-up" (Unity), "unreal" (Unreal)

    Returns:
        points_world: [N, 3] points in world coordinates
    """
    points_world = points_cam.copy()

    # Undo SAM3D's Y/Z flip
    points_world[:, 1] *= -1  # Flip Y back
    points_world[:, 2] *= -1  # Flip Z back

    if target_system == "y-up":
        # Blender/OpenGL: Y-up, Z-forward
        # Already correct after undoing flip
        pass
    elif target_system == "z-up":
        # Unity/many games: Z-up, Y-forward
        x, y, z = points_world[:, 0], points_world[:, 1], points_world[:, 2]
        points_world[:, 0] = x
        points_world[:, 1] = -z  # Depth becomes Y
        points_world[:, 2] = -y  # Height becomes Z
    elif target_system == "unreal":
        # Unreal: Z-up, X-forward, Y-right
        x, y, z = points_world[:, 0], points_world[:, 1], points_world[:, 2]
        points_world[:, 0] = y    # Y becomes X (forward)
        points_world[:, 1] = -x   # -X becomes Y (right)
        points_world[:, 2] = z    # Z stays Z (up)

    return points_world

# Convert mesh and keypoints
vertices_world = sam3d_to_world_coords(vertices_cam, "y-up")
keypoints_world = sam3d_to_world_coords(keypoints_cam, "y-up")
```

#### Step 4: Apply Global Rotation

If you want the person to face a specific direction in your scene:

```python
def apply_global_rotation(points, yaw_deg=0, pitch_deg=0, roll_deg=0):
    """
    Apply rotation around world axes.

    Args:
        points: [N, 3] points to rotate
        yaw_deg: rotation around Y-axis (degrees)
        pitch_deg: rotation around X-axis (degrees)
        roll_deg: rotation around Z-axis (degrees)
    """
    import scipy.spatial.transform as R

    # Create rotation matrix (ZYX order)
    rotation = R.from_euler('zyx', [yaw_deg, pitch_deg, roll_deg], degrees=True)
    rot_matrix = rotation.as_matrix()

    # Center at origin first
    centroid = points.mean(axis=0)
    points_centered = points - centroid

    # Apply rotation
    points_rotated = points_centered @ rot_matrix.T

    # Return to original position
    return points_rotated + centroid

# Optional: Override detected rotation with custom orientation
vertices_world = apply_global_rotation(vertices_world, yaw_deg=45)
keypoints_world = apply_global_rotation(keypoints_world, yaw_deg=45)
```

#### Step 5: Position in Your Scene

Place the character at the desired world position:

```python
def place_in_scene(points, world_position, scale=1.0):
    """
    Place mesh at specific world position with optional scaling.

    Args:
        points: [N, 3] points in world coordinates
        world_position: [x, y, z] target position
        scale: uniform scale factor
    """
    # Apply scaling
    if scale != 1.0:
        centroid = points.mean(axis=0)
        points = (points - centroid) * scale + centroid

    # Translate to world position
    return points + np.array(world_position)

# Place person at world coordinates (10, 0, 5)
final_vertices = place_in_scene(vertices_world, [10, 0, 5])
final_keypoints = place_in_scene(keypoints_world, [10, 0, 5])
```

#### Step 6: Ground the Character

Ensure the character stands on the ground plane:

```python
def ground_character(vertices, keypoints, ground_axis=2):
    """
    Move character so lowest point touches ground.

    Args:
        vertices: [N, 3] mesh vertices
        keypoints: [70, 3] keypoints
        ground_axis: axis to align with ground (0=X, 1=Y, 2=Z)
    """
    # Find lowest point
    min_height = min(vertices[:, ground_axis].min(), keypoints[:, ground_axis].min())

    # Shift so ground is at 0
    vertices[:, ground_axis] -= min_height
    keypoints[:, ground_axis] -= min_height

    return vertices, keypoints

# Apply grounding
final_vertices, final_keypoints = ground_character(final_vertices, final_keypoints)
```

#### Step 7: Export for Your 3D Engine

```python
def export_for_engine(vertices, faces, keypoints, engine="blender"):
    """Export in format suitable for different 3D engines."""

    if engine == "blender":
        # Export as OBJ
        export_to_obj(vertices, faces, "person.obj", keypoints=keypoints)

    elif engine == "unity":
        # Export as FBX with rig
        export_to_fbx(vertices, faces, "person.fbx", keypoints=keypoints)

    elif engine == "unreal":
        # Export as FBX with UE4 skeleton
        export_unreal_fbx(vertices, faces, "person_ue.fbx", keypoints=keypoints)

    elif engine == "threejs":
        # Export as JSON for Three.js
        export_threejs_json(vertices, faces, "person.json", keypoints=keypoints)

# Export the final mesh
export_for_engine(final_vertices, output["faces"], final_keypoints, "blender")
```

#### Complete Integration Example

```python
def process_and_place_person(image_path, world_pos=[0, 0, 0], world_rotation=0):
    """Complete pipeline from image to 3D scene placement."""

    # 1. Run inference
    img = cv2.imread(image_path)
    outputs = estimator.process_one_image(img)
    output = outputs[0]

    # 2. Extract data
    vertices = output["pred_vertices"]
    keypoints = output["pred_keypoints_3d"]
    cam_t = output["pred_cam_t"]

    # 3. Camera frame
    vertices_cam, keypoints_cam = to_camera_frame(vertices, keypoints, cam_t)

    # 4. World coordinates
    vertices_world = sam3d_to_world_coords(vertices_cam, "y-up")
    keypoints_world = sam3d_to_world_coords(keypoints_cam, "y-up")

    # 5. Apply rotation
    vertices_world = apply_global_rotation(vertices_world, yaw_deg=world_rotation)
    keypoints_world = apply_global_rotation(keypoints_world, yaw_deg=world_rotation)

    # 6. Place in scene
    final_vertices = place_in_scene(vertices_world, world_pos)
    final_keypoints = place_in_scene(keypoints_world, world_pos)

    # 7. Ground
    final_vertices, final_keypoints = ground_character(final_vertices, final_keypoints)

    # 8. Export
    export_for_engine(final_vertices, output["faces"], final_keypoints, "blender")

    return {
        "vertices": final_vertices,
        "keypoints": final_keypoints,
        "position": world_pos,
        "rotation": world_rotation
    }

# Usage
result = process_and_place_person(
    "person.jpg",
    world_pos=[5, 0, 10],  # Place at x=5, z=10 in scene
    world_rotation=30       # Rotate 30 degrees
)
```

#### Quick Verification Script

To verify the placement is correct:

```python
def verify_placement(keypoints_3d, output):
    """Verify 3D placement against original 2D keypoints."""

    # Extract relevant info from output
    focal = output["focal_length"]
    cam_t = output["pred_cam_t"]
    kp_2d_pred = output["pred_keypoints_2d"]

    # Project 3D keypoints back to 2D
    kp_2d_proj = []
    for kp in keypoints_3d:
        if kp[2] > 0.1:  # Not behind camera
            x = focal * kp[0] / kp[2] + 512  # Assuming 1024x1024 image
            y = focal * kp[1] / kp[2] + 384
            kp_2d_proj.append([x, y])

    # Compare with model's 2D predictions
    kp_2d_proj = np.array(kp_2d_proj)
    kp_2d_pred = np.array(kp_2d_pred)

    # Calculate reprojection error
    error = np.linalg.norm(kp_2d_proj - kp_2d_pred, axis=1)
    print(f"Mean reprojection error: {error.mean():.2f} pixels")
    print(f"Max reprojection error: {error.max():.2f} pixels")

    return error.mean() < 5.0  # Good if error < 5 pixels

# Verify
verify_placement(final_keypoints, output)
```

This complete workflow takes you from a single image to a properly positioned 3D character in your scene!

---

### Working with Joint Rotations

#### Getting Joint Orientations for Animation

```python
def get_joint_transforms(output):
    """Extract joint positions and orientations for rigging/animation."""

    joint_positions = output["pred_joint_coords"]  # [127, 3] in meters
    joint_rotations = output["pred_global_rots"]   # [127, 3, 3] rotation matrices

    transforms = []
    for i in range(127):
        transform = {
            "joint_idx": i,
            "position": joint_positions[i],  # [3]
            "rotation_matrix": joint_rotations[i],  # [3, 3]
        }

        # Convert to quaternion if needed
        import scipy.spatial.transform as R
        rot = R.Rotation.from_matrix(joint_rotations[i])
        transform["quaternion"] = rot.as_quat()  # [x, y, z, w]
        transform["euler_xyz"] = rot.as_euler('xyz', degrees=True)

        transforms.append(transform)

    return transforms
```

#### Applying to a Different Skeleton

If you have a target skeleton (e.g., Mixamo, UE5 Mannequin), you need to:

1. **Retarget joint mapping**: Map MHR joints to your skeleton's joints
2. **Handle different rest poses**: MHR's T-pose may differ from your skeleton
3. **Apply local rotations**: Use `body_pose_params` for local joint angles

```python
# Example joint mapping (MHR → Mixamo approximate)
MHR_TO_MIXAMO = {
    0: "Hips",
    1: "Spine",
    2: "Spine1",
    3: "Spine2",
    4: "Neck",
    5: "Head",
    # ... etc
}
```

---

### Rendering the Mesh

#### Using PyRender (as in SAM 3D Body)

```python
import numpy as np
import pyrender
import trimesh

def render_mesh(vertices, faces, cam_t, focal_length, image_size):
    """Render mesh overlay on image."""

    # Create trimesh
    mesh = trimesh.Trimesh(vertices + cam_t, faces)

    # Flip for rendering (SAM3D convention)
    rot = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    mesh.apply_transform(rot)

    # Create pyrender mesh
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        baseColorFactor=(0.65, 0.74, 0.86, 1.0)
    )
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material)

    # Create scene
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh_pyrender)

    # Camera
    camera = pyrender.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=image_size[0]/2, cy=image_size[1]/2
    )
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 0, 0]  # Camera at origin
    camera_pose[0, 3] *= -1  # Flip X for OpenGL
    scene.add(camera, pose=camera_pose)

    # Render
    renderer = pyrender.OffscreenRenderer(image_size[0], image_size[1])
    color, depth = renderer.render(scene)
    renderer.delete()

    return color, depth
```

---

### Practical Tips

#### 1. Typical Output Ranges
```python
# pred_cam_t[2] (depth): typically 2-10 meters
# pred_vertices: typically within [-1, 1] meters from root
# focal_length: typically 500-2000 pixels for consumer cameras
```

#### 2. Checking Output Validity
```python
def validate_output(output):
    """Quick sanity checks on model output."""

    depth = output["pred_cam_t"][2]
    if depth < 0.5 or depth > 20:
        print(f"Warning: Unusual depth {depth:.2f}m")

    # Check mesh bounds
    verts = output["pred_vertices"]
    extent = verts.max(axis=0) - verts.min(axis=0)
    if extent.max() > 3.0:  # Person shouldn't be > 3m in any dimension
        print(f"Warning: Large mesh extent {extent}")

    # Check for NaN/Inf
    if np.any(np.isnan(verts)) or np.any(np.isinf(verts)):
        print("Warning: Invalid values in vertices")
        return False

    return True
```

#### 3. Multi-Person Scenes
```python
# When processing multiple people, they're all in the same camera frame
outputs = estimator.process_one_image(image)

# Each person has their own cam_t, so they're correctly positioned
# relative to each other
all_vertices = []
for i, out in enumerate(outputs):
    verts = out["pred_vertices"] + out["pred_cam_t"]
    all_vertices.append(verts)

# Now all_vertices contains meshes positioned correctly in camera space
```

#### 4. Converting Depth to Real-World Scale

If you know the actual size of something in the scene (e.g., a known object):
```python
def rescale_to_real_world(output, known_height_m, keypoint_indices=(13, 5)):
    """Rescale using known height (e.g., person height from ankle to shoulder)."""

    kps = output["pred_keypoints_3d"]
    measured_height = np.linalg.norm(kps[keypoint_indices[0]] - kps[keypoint_indices[1]])

    scale_factor = known_height_m / measured_height

    output["pred_vertices"] *= scale_factor
    output["pred_keypoints_3d"] *= scale_factor
    output["pred_cam_t"] *= scale_factor

    return output, scale_factor
```

---

### Summary Table

| Output | Shape | Units | Description |
|--------|-------|-------|-------------|
| `pred_vertices` | [18439, 3] | meters | Mesh vertices (root-relative) |
| `pred_keypoints_3d` | [70, 3] | meters | Body landmarks |
| `pred_keypoints_2d` | [70, 2] | pixels | Projected 2D keypoints |
| `pred_cam_t` | [3] | meters | Camera translation (tx, ty, depth) |
| `pred_joint_coords` | [127, 3] | meters | Skeleton joint positions |
| `pred_global_rots` | [127, 3, 3] | - | Joint rotation matrices |
| `global_rot` | [3] | radians | Body orientation (ZYX euler) |
| `body_pose_params` | [130] | radians | Local joint angles |
| `shape_params` | [45] | - | Body shape PCA coefficients |
| `scale_params` | [28] | - | Bone proportion PCA coefficients |
| `hand_pose_params` | [108] | - | Hand pose PCA (54 per hand) |
| `focal_length` | scalar | pixels | Camera focal length |

This gives you everything needed to place SAM 3D Body outputs into any 3D scene or animation system.
