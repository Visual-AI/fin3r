# Source notices

The model and training code build on Meta VGGT and the Fin3R QKV adaptation. The release includes the VGGT license and the original notices for bundled components. The base model and training datasets are obtained from their respective providers.

| Component | Upstream identity and license |
|---|---|
| VGGT backbone | [facebookresearch/vggt](https://github.com/facebookresearch/vggt). [VGGT License](LICENSE.txt) |
| Fin3R adaptation | [Visual-AI/Fin3R](https://github.com/Visual-AI/Fin3R), reference revision `09c087489133b7e90fd0a86612b0c4b06986c289` |
| MoGe teacher and depth metrics | [microsoft/MoGe](https://github.com/microsoft/MoGe), revision `74fbce054ebed49800de42d0ad0e83495065719a`. [Notices](third_party/moge/LICENSE) |
| utils3d | Revision `62f09d58509485564e24d5d9f6aac9ee9ebc0c37`. [License](third_party/utils3d/LICENSE) |
| Pi3 evaluation | Evaluation revision `97ce97b8e067e7bff5b5c8e507776e4e95189fac`. The bundled Pi3 code carries its non-commercial license. [License](third_party/pi3/LICENSE) |
| pipeline | [License](third_party/pipeline/LICENSE) |

The adapter, original source commit and training schedule identities are recorded in [release.json](release.json). The distributed schedule preserves sample IDs, frame order and rank assignments and uses paths relative to the user's data directory.
