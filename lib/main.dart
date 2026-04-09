import 'dart:math';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

void main() => runApp(const OryzaApp());

class OryzaApp extends StatelessWidget {
  const OryzaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'OryzaScanner',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.green,
        useMaterial3: true,
      ),
      home: const MainNavigation(),
    );
  }
}

class MainNavigation extends StatefulWidget {
  const MainNavigation({super.key});

  @override
  State<MainNavigation> createState() => _MainNavigationState();
}

class _MainNavigationState extends State<MainNavigation> {
  int _selectedIndex = 0;
  final List<Widget> _pages = [
    const NutrientScanner(),
    const Scanner2(),
    const AboutPage(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _selectedIndex,
        children: _pages,
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (index) => setState(() => _selectedIndex = index),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.radar), label: 'Detect'),
          NavigationDestination(icon: Icon(Icons.center_focus_strong), label: 'Analyze'),
          NavigationDestination(icon: Icon(Icons.info_outline), label: 'About'),
        ],
      ),
    );
  }
}

class NutrientScanner extends StatefulWidget {
  const NutrientScanner({super.key});
  @override
  State<NutrientScanner> createState() => _NutrientScannerState();
}

class _NutrientScannerState extends State<NutrientScanner> with AutomaticKeepAliveClientMixin {
  @override
  bool get wantKeepAlive => true;

  bool _isLoading = false;
  File? _image;
  String _result = "Waiting for Scan...";
  String _solution = "Capture or upload a photo of the paddy leaf to begin.";
  Interpreter? yolo;
  Interpreter? classifier;
  List<Map<String, dynamic>> _finalDetections = [];
  Size? _originalImageSize;

  final Map<String, String> _solutionBook = {
    "Nitrogen Deficiency": "Soil application of urea @ 330/ha (110/ ac at three times). Foliar application of urea 1% (10 g/lit) at 15 days interval till the symptoms disappear.",
    "Iron Deficiency": "Apply ferrous sulphate @ 0.5 % (5g/lit) as foliar spray, two times at 15 days interval.",
    "Healthy": "Plant looks good! Continue regular monitoring and balanced irrigation.",
  };

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      var options = InterpreterOptions()..threads = 4;
      classifier = await Interpreter.fromAsset('assets/model.tflite', options: options);
      yolo = await Interpreter.fromAsset('assets/best_float32.tflite', options: options);
    } catch (e) {
      print("❌ Model load error: $e");
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _isLoading = true;
        _finalDetections = [];
      });
      _processImage(_image!);
    }
  }

  Future<void> _processImage(File file) async {
    if (classifier == null || yolo == null) {
      _showError("Model Error", "Models not loaded correctly.");
      return;
    }
    setState(() {
      _isLoading = true;
      _finalDetections = [];
      _result = "Analyzing...";
    });

    try {
      final bytes = await file.readAsBytes();
      img.Image? original = await compute(img.decodeImage, bytes);
      if (original == null) return;

      _originalImageSize = Size(original.width.toDouble(), original.height.toDouble());

      // 1. YOLO Detection
      img.Image resizedYolo = img.copyResize(original, width: 832, height: 832);
      var input = Float32List(1 * 832 * 832 * 3);
      int pixelIdx = 0;
      for (var pixel in resizedYolo) {
        input[pixelIdx++] = pixel.r / 255.0;
        input[pixelIdx++] = pixel.g / 255.0;
        input[pixelIdx++] = pixel.b / 255.0;
      }
      var output = List.filled(1 * 5 * 14196, 0.0).reshape([1, 5, 14196]);
      yolo!.run(input.reshape([1, 832, 832, 3]), output);

      // 2. NMS with Threshold (0.45 threshold to filter non-leaves)
      List<Map<String, dynamic>> finalBoxes = _performNMS(output[0], original);

      if (finalBoxes.isEmpty) {
        _showError("No plant detected ❌", "Please upload a clearer photo of a leaf.");
        return;
      }

      // 3. Crop and Classify Loop
      Map<String, int> counts = {};
      for (var box in finalBoxes) {
        img.Image croppedLeaf = img.copyCrop(
            original,
            x: box["x1"],
            y: box["y1"],
            width: box["x2"] - box["x1"],
            height: box["y2"] - box["y1"]
        );

        String labelResult = _classifyLeaf(croppedLeaf).split("(")[0].trim();
        box["label"] = labelResult;
        counts[labelResult] = (counts[labelResult] ?? 0) + 1;
      }

      setState(() {
        _finalDetections = finalBoxes;
        _isLoading = false;
        _result = "Analysis Complete";
        _solution = _getRecommendation(counts); // Trigger Combined Action if both exist
      });
    } catch (e) {
      _showError("Error", "Analysis failed: $e");
    }
  }

  List<Map<String, dynamic>> _performNMS(List<List<double>> outputData, img.Image original) {
    List<Map<String, dynamic>> detections = [];
    double threshold = 0.45; // THRESHOLD TO AVOID NON-LEAVES

    for (int i = 0; i < 14196; i++) {
      double conf = outputData[4][i];
      if (conf > threshold) {
        double x = outputData[0][i], y = outputData[1][i], w = outputData[2][i], h = outputData[3][i];
        int x1 = ((x - w / 2) * original.width).toInt().clamp(0, original.width);
        int y1 = ((y - h / 2) * original.height).toInt().clamp(0, original.height);
        int x2 = ((x + w / 2) * original.width).toInt().clamp(0, original.width);
        int y2 = ((y + h / 2) * original.height).toInt().clamp(0, original.height);
        detections.add({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf});
      }
    }
    detections.sort((a, b) => b["conf"].compareTo(a["conf"]));
    List<Map<String, dynamic>> finalBoxes = [];
    for (var det in detections) {
      bool keep = true;
      for (var fb in finalBoxes) {
        if (_calculateIoU(det, fb) > 0.3) {
          keep = false;
          break;
        }
      }
      if (keep) finalBoxes.add(det);
      if (finalBoxes.length >= 20) break;
    }
    return finalBoxes;
  }

  String _getRecommendation(Map<String, int> counts) {
    bool hasNitrogen = counts.containsKey("Nitrogen Deficiency");
    bool hasIron = counts.containsKey("Iron Deficiency");

    if (hasNitrogen && hasIron) {
      return "COMBINED ACTION:\n1. NITROGEN: ${_solutionBook["Nitrogen Deficiency"]}\n\n2. IRON: ${_solutionBook["Iron Deficiency"]}";
    } else if (hasNitrogen) {
      return _solutionBook["Nitrogen Deficiency"]!;
    } else if (hasIron) {
      return _solutionBook["Iron Deficiency"]!;
    } else {
      return _solutionBook["Healthy"]!;
    }
  }

  void _showError(String title, String message) {
    setState(() {
      _isLoading = false;
      _result = title;
      _solution = message;
      _finalDetections = [];
    });
  }

  double _calculateIoU(Map a, Map b) {
    int x1 = max(a["x1"], b["x1"]), y1 = max(a["y1"], b["y1"]), x2 = min(a["x2"], b["x2"]), y2 = min(a["y2"], b["y2"]);
    int interArea = max(0, x2 - x1) * max(0, y2 - y1);
    int areaA = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]), areaB = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]);
    return interArea / (areaA + areaB - interArea + 1e-6);
  }

  String _classifyLeaf(img.Image leaf) {
    img.Image resized = img.copyResize(leaf, width: 224, height: 224, maintainAspect: true);
    var input = Float32List(1 * 224 * 224 * 3);
    int idx = 0;
    for (var pixel in resized) {
      input[idx++] = pixel.r / 255.0;
      input[idx++] = pixel.g / 255.0;
      input[idx++] = pixel.b / 255.0;
    }
    var output = List.filled(3, 0.0).reshape([1, 3]);
    classifier!.run(input.reshape([1, 224, 224, 3]), output);

    // CRITICAL: MUST MATCH PYTHON ORDER [Nitrogen, Iron, Healthy]
    // Try alphabetical order if the results seem swapped
    List<String> labels = ["Nitrogen Deficiency", "Iron Deficiency", "Healthy"];
    int maxIdx = 0;
    for (int i = 1; i < 3; i++) if (output[0][i] > output[0][maxIdx]) maxIdx = i;

    return "${labels[maxIdx]} (${(output[0][maxIdx] * 100).toStringAsFixed(1)}%)";
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text("OryzaScanner", style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white)),
        backgroundColor: Colors.green[800],
        centerTitle: true,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            const SizedBox(height: 15),
            _buildImageContainer(),
            const SizedBox(height: 15),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildActionButton(Icons.camera_alt, "CAMERA", Colors.green[700]!, () => _pickImage(ImageSource.camera)),
                _buildActionButton(Icons.photo_library, "GALLERY", Colors.blueGrey[700]!, () => _pickImage(ImageSource.gallery)),
              ],
            ),
            const SizedBox(height: 15),
            _buildResultCard(),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  Widget _buildImageContainer() {
    return Container(
      height: 300,
      width: double.infinity,
      margin: const EdgeInsets.symmetric(horizontal: 20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(25),
        border: Border.all(color: Colors.green.withOpacity(0.3), width: 2),
        boxShadow: [BoxShadow(color: Colors.green.withOpacity(0.05), blurRadius: 15, spreadRadius: 2)],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(23),
        child: Stack(
          children: [
            Center(child: Icon(Icons.eco, size: 100, color: Colors.green.withOpacity(0.05))),
            if (_isLoading)
              const Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(color: Colors.green, strokeWidth: 3),
                    SizedBox(height: 15),
                    Text("Analyzing Leaf...", style: TextStyle(color: Colors.green, fontWeight: FontWeight.bold)),
                  ],
                ),
              )
            else if (_image == null)
              const Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.add_a_photo_outlined, size: 40, color: Colors.green),
                    SizedBox(height: 8),
                    Text("Ready for Inspection", style: TextStyle(color: Colors.grey, fontWeight: FontWeight.w500)),
                  ],
                ),
              )
            else
              LayoutBuilder(
                builder: (context, constraints) {
                  return Stack(
                    fit: StackFit.expand,
                    children: [
                      Image.file(_image!, fit: BoxFit.contain),
                      if (_originalImageSize != null)
                        CustomPaint(
                          size: Size(constraints.maxWidth, constraints.maxHeight),
                          painter: BoxPainter(_finalDetections, _originalImageSize!),
                        ),
                    ],
                  );
                },
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard() {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 20),
      elevation: 4,
      shadowColor: Colors.black12,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text("DIAGNOSTIC SUMMARY", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12, color: Colors.blueGrey, letterSpacing: 1.1)),
            const SizedBox(height: 15),
            Row(
              children: [
                Icon(Icons.layers_outlined, color: Colors.green[700]),
                const SizedBox(width: 8),
                Text("${_finalDetections.length} Leaves Detected", style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 15),
            Wrap(spacing: 8, runSpacing: 8, children: _buildResultChips()),
            const Padding(padding: EdgeInsets.symmetric(vertical: 15), child: Divider()),
            const Text("RECOMMENDED ACTION", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12, color: Colors.blueGrey, letterSpacing: 1.1)),
            const SizedBox(height: 10),
            Container(
              padding: const EdgeInsets.all(12),
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.green[50],
                borderRadius: BorderRadius.circular(10),
              ),
              child: Text(
                  _solution,
                  style: const TextStyle(fontSize: 14, color: Colors.black87, height: 1.4)
              ),
            ),
          ],
        ),
      ),
    );
  }

  List<Widget> _buildResultChips() {
    if (_finalDetections.isEmpty) {
      return [const Text("Waiting for Scan...", style: TextStyle(color: Colors.grey, fontStyle: FontStyle.italic))];
    }
    Map<String, int> counts = {};
    for (var det in _finalDetections) {
      String label = det['label'] ?? "Unknown";
      counts[label] = (counts[label] ?? 0) + 1;
    }
    return counts.entries.map((entry) {
      Color chipColor = entry.key == "Nitrogen Deficiency" ? Colors.red[100]! : (entry.key == "Iron Deficiency" ? Colors.orange[100]! : Colors.green[100]!);
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(color: chipColor, borderRadius: BorderRadius.circular(20), border: Border.all(color: Colors.black12)),
        child: Text("${entry.key}: ${entry.value}", style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12)),
      );
    }).toList();
  }

  Widget _buildActionButton(IconData icon, String label, Color color, VoidCallback onTap) {
    return ElevatedButton.icon(
      onPressed: onTap,
      icon: Icon(icon, color: Colors.white),
      label: Text(label, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
      style: ElevatedButton.styleFrom(backgroundColor: color, minimumSize: const Size(140, 50), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12))),
    );
  }
}

class Scanner2 extends StatefulWidget {
  const Scanner2({super.key});
  @override
  State<Scanner2> createState() => _Scanner2State();
}

class _Scanner2State extends State<Scanner2> with AutomaticKeepAliveClientMixin {
  @override
  bool get wantKeepAlive => true;

  bool _isLoading = false;
  File? _image;
  String _result = "Ready to Scan";
  String _solution = "Upload a leaf photo for direct AI analysis.";
  Interpreter? classifier;

  final Map<String, String> _solutionBook = {
    "Nitrogen Deficiency": "Soil application of urea @ 330/ha (110/ ac at three times). Foliar application of urea 1% (10 g/lit) at 15 days interval till the symptoms disappear.",
    "Iron Deficiency": "Apply ferrous sulphate @ 0.5 % (5g/lit) as foliar spray, two times at 15 days interval.",
    "Healthy": "Plant looks good! Continue regular monitoring and balanced irrigation.",
  };

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      var options = InterpreterOptions()..threads = 4;
      classifier = await Interpreter.fromAsset('assets/model.tflite', options: options);
    } catch (e) {
      print("❌ Model load error: $e");
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() { _image = File(pickedFile.path); _isLoading = true; });
      _processImage(_image!);
    }
  }

  Future<void> _processImage(File file) async {
    if (classifier == null) return;
    setState(() => _isLoading = true);

    try {
      final bytes = await file.readAsBytes();
      img.Image? original = await compute(img.decodeImage, bytes);
      if (original == null) return;

      // 1. Pre-processing: Ensure 224x224 and Normalize
      img.Image resized = img.copyResize(original, width: 224, height: 224);
      var input = Float32List(1 * 224 * 224 * 3);
      int idx = 0;
      for (var y = 0; y < 224; y++) {
        for (var x = 0; x < 224; x++) {
          var pixel = resized.getPixel(x, y);
          input[idx++] = pixel.r / 255.0;
          input[idx++] = pixel.g / 255.0;
          input[idx++] = pixel.b / 255.0;
        }
      }

      // 2. Run Inference
      var output = List.filled(3, 0.0).reshape([1, 3]);
      classifier!.run(input.reshape([1, 224, 224, 3]), output);

      // 3. Label Mapping & Sensitivity Adjustment
      // CRITICAL: Ensure this matches your Python/Training export order exactly.
      // Common order: [Healthy, Iron, Nitrogen]
      List<String> labels = ["Nitrogen Deficiency", "Iron Deficiency", "Healthy"];

      List<double> scores = List<double>.from(output[0]);

      // OPTIONAL: Sensitivity Bias
      // If Nitrogen is subtle, we can apply a small weight to its score
      // to prevent it from being "drowned out" by the Healthy class.
      // scores[2] = scores[2] * 1.2;

      int maxIdx = 0;
      double maxScore = -1.0;

      for (int i = 0; i < scores.length; i++) {
        if (scores[i] > maxScore) {
          maxScore = scores[i];
          maxIdx = i;
        }
      }

      // 4. Final Threshold Check
      // If the top result is "Healthy" but Nitrogen is above a 0.3 threshold,
      // you might want to flag it as a warning instead.
      String finalLabel = labels[maxIdx];
      if (maxScore < 0.40) {
        finalLabel = "Inconclusive - Retake Photo";
      }

      setState(() {
        _isLoading = false;
        _result = "$finalLabel (${(maxScore * 100).toStringAsFixed(1)}%)";
        _solution = _solutionBook[finalLabel] ?? "Please try a clearer image of the leaf.";
      });

    } catch (e) {
      setState(() {
        _isLoading = false;
        _result = "Analysis Error";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text("OryzaScanner", style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white)),
        backgroundColor: Colors.green[800],
        centerTitle: true,
        elevation: 0,
      ),
      body: Column(
        children: [
          const SizedBox(height: 15),
          Container(
            height: 300,
            width: double.infinity,
            margin: const EdgeInsets.symmetric(horizontal: 20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(25),
              border: Border.all(color: Colors.green.withOpacity(0.3), width: 2),
              boxShadow: [BoxShadow(color: Colors.green.withOpacity(0.05), blurRadius: 15, spreadRadius: 2)],
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(23),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  if (_image == null)
                    Stack(
                      alignment: Alignment.center,
                      children: [
                        Icon(Icons.eco, size: 100, color: Colors.green.withOpacity(0.05)),
                        Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: const [
                              Icon(Icons.add_a_photo_outlined, size: 40, color: Colors.green),
                              SizedBox(height: 8),
                              Text("Ready for Inspection", style: TextStyle(color: Colors.grey, fontWeight: FontWeight.w500)),
                            ],
                          ),
                        ),
                      ],
                    )
                  else
                    Image.file(_image!, fit: BoxFit.contain),
                  if (_isLoading) const Center(child: CircularProgressIndicator(color: Colors.green)),
                ],
              ),
            ),
          ),
          const SizedBox(height: 15),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _buildBtn(Icons.camera_alt, "CAMERA", Colors.green[700]!, () => _pickImage(ImageSource.camera)),
              _buildBtn(Icons.photo_library, "GALLERY", Colors.blueGrey[700]!, () => _pickImage(ImageSource.gallery)),
            ],
          ),
          const SizedBox(height: 15),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 20),
            elevation: 4,
            shadowColor: Colors.black12,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text("DIAGNOSTIC SUMMARY", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12, color: Colors.blueGrey, letterSpacing: 1.1)),
                  const SizedBox(height: 15),
                  Text(_result, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                  const Padding(padding: EdgeInsets.symmetric(vertical: 15), child: Divider()),
                  const Text("RECOMMENDED ACTION", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12, color: Colors.blueGrey, letterSpacing: 1.1)),
                  const SizedBox(height: 10),
                  Container(
                    padding: const EdgeInsets.all(12),
                    width: double.infinity,
                    decoration: BoxDecoration(color: Colors.green[50], borderRadius: BorderRadius.circular(10)),
                    child: Text(_solution, style: const TextStyle(fontSize: 14, color: Colors.black87, height: 1.4)),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBtn(IconData icon, String label, Color color, VoidCallback onTap) {
    return ElevatedButton.icon(
      onPressed: onTap,
      icon: Icon(icon, color: Colors.white),
      label: Text(label, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
      style: ElevatedButton.styleFrom(backgroundColor: color, minimumSize: const Size(140, 50), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12))),
    );
  }
}

class BoxPainter extends CustomPainter {
  final List<Map<String, dynamic>> detections;
  final Size originalSize;
  BoxPainter(this.detections, this.originalSize);
  @override
  void paint(Canvas canvas, Size size) {
    if (detections.isEmpty) return;
    double scale = min(size.width / originalSize.width, size.height / originalSize.height);
    double offsetX = (size.width - originalSize.width * scale) / 2;
    double offsetY = (size.height - originalSize.height * scale) / 2;
    for (var det in detections) {
      Color boxColor = det['label'] == "Nitrogen Deficiency" ? Colors.red : (det['label'] == "Iron Deficiency" ? Colors.yellow : Colors.greenAccent);
      final paint = Paint()..color = boxColor..style = PaintingStyle.stroke..strokeWidth = 3.0;
      double left = det['x1'] * scale + offsetX, top = det['y1'] * scale + offsetY, right = det['x2'] * scale + offsetX, bottom = det['y2'] * scale + offsetY;
      canvas.drawRect(Rect.fromLTRB(left, top, right, bottom), paint);
      TextSpan span = TextSpan(style: TextStyle(color: boxColor == Colors.yellow ? Colors.black : Colors.white, fontSize: 10, fontWeight: FontWeight.bold), text: " ${det['label']} ");
      TextPainter tp = TextPainter(text: span, textDirection: TextDirection.ltr)..layout();
      canvas.drawRRect(RRect.fromRectAndRadius(Rect.fromLTWH(left, top - 18, tp.width + 4, 18), const Radius.circular(5)), Paint()..color = boxColor);
      tp.paint(canvas, Offset(left + 2, top - 16));
    }
  }
  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}

class AboutPage extends StatelessWidget {
  const AboutPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text("System Information", style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white)),
        backgroundColor: Colors.green[800],
        centerTitle: true,
        elevation: 0,
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 30.0),
        child: Column(
          children: [
            Expanded(
              flex: 3,
              child: Container(
                decoration: BoxDecoration(color: Colors.green[50], shape: BoxShape.circle),
                child: Center(child: Icon(Icons.psychology, size: 80, color: Colors.green[700])),
              ),
            ),
            const SizedBox(height: 20),
            const Text("OryzaScanner AI", style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: Colors.black87)),
            const Text("Version 2.5.0", style: TextStyle(color: Colors.grey, fontWeight: FontWeight.w500)),
            const Spacer(flex: 1),
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(color: Colors.grey[50], borderRadius: BorderRadius.circular(20), border: Border.all(color: Colors.grey[200]!)),
              child: Column(
                children: [
                  _buildTechRow(Icons.grid_view_rounded, "Detection Engine", "YOLO Architecture"),
                  const Divider(height: 30),
                  _buildTechRow(Icons.settings_input_component, "Model Type", "TFLite Neural Network"),
                  const Divider(height: 30),
                  _buildTechRow(Icons.speed, "Processing", "Real-time AI Inference"),
                ],
              ),
            ),
            const Spacer(flex: 2),
            const Text("© 2026 OryzaScanner AI Technologies", style: TextStyle(color: Colors.grey, fontSize: 12, letterSpacing: 0.5)),
          ],
        ),
      ),
    );
  }

  Widget _buildTechRow(IconData icon, String title, String value) {
    return Row(
      children: [
        Icon(icon, color: Colors.green[700], size: 28),
        const SizedBox(width: 15),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: const TextStyle(fontSize: 14, color: Colors.blueGrey, fontWeight: FontWeight.w600)),
            Text(value, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black87)),
          ],
        ),
      ],
    );
  }
}