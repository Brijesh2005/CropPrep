# Introduction

CropFusion is a multimodal pipeline that couples environmental tabular
descriptors with satellite imagery to classify crops at field level in
Karnataka's Dakshina Kannada district. The central research question of this
final experiment is whether **CropFusion (tabular + imagery)** provides a
detectable advantage over **tabular-only** and **imagery-only** models.

The agricultural setting is demanding: black pepper is pervasively
intercropped under coconut canopies at short distances, so the two crops share
GPS coordinates, environment and satellite pixels. Earlier experiments
(R5.5-R5.10) consistently recorded ~50% balanced accuracy regardless of
representation, motivating a strictly controlled final comparison.
