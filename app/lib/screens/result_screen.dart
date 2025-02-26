import 'dart:convert';
import 'dart:typed_data';
import 'package:boulder_ai/util/detection_overlay.dart';
import 'package:boulder_ai/util/image_gallery.dart';
import 'package:boulder_ai/util/util.dart';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class ResultScreen extends StatefulWidget {
  final Map<String, dynamic> processedResult;

  const ResultScreen({super.key, required this.processedResult});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  int? selectedRouteIndex;
  bool _isRoutesPanelOpen = true;
  @override
  Widget build(BuildContext context) {
    final storage = const FlutterSecureStorage();
    const double distanceThreshold = 320.0;

    return FutureBuilder<String?>(
      future: storage.read(key: 'jwt_token'),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return Scaffold(
            backgroundColor: const Color(0xFF1C1C1E),
            body: const Center(child: CircularProgressIndicator()),
          );
        }
        if (snapshot.data == null) {
          WidgetsBinding.instance.addPostFrameCallback((_) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text(
                  "Deine Sitzung ist abgelaufen. Bitte logge dich erneut ein.",
                  style: TextStyle(color: Colors.white),
                ),
                backgroundColor: Colors.red,
                duration: Duration(seconds: 3),
              ),
            );
            Navigator.pushReplacementNamed(context, '/login');
          });
          return Scaffold(
            backgroundColor: const Color(0xFF1C1C1E),
            body: const Center(child: Text("Leite um...")),
          );
        }

        final Color backgroundColor = const Color(0xFF1C1C1E);
        final Color cardColor = const Color(0xFF2C2C2E);
        final Color textColor = Colors.white;

        Map<int, Color> classColorMap = {
          0: Colors.black,
          1: Colors.blue,
          2: Colors.grey,
          3: Colors.orange,
          4: Colors.purple,
          5: Colors.red,
          6: Colors.cyan,
          7: Colors.white,
          8: const Color(0xFF8D6E63),
          9: Colors.yellow,
        };

        final List<String> classNames = [
          "black",
          "blue",
          "grey",
          "orange",
          "purple",
          "red",
          "turquoise",
          "white",
          "wood",
          "yellow",
        ];

        Uint8List origBytes;
        Uint8List detectBytes;
        try {
          final String origBase64 =
              widget.processedResult["original_image"] as String;
          final String detectBase64 =
              widget.processedResult["detection_image"] as String;
          origBytes = base64Decode(origBase64);
          detectBytes = base64Decode(detectBase64);
        } catch (e) {
          return Scaffold(
            backgroundColor: backgroundColor,
            appBar: AppBar(
              backgroundColor: backgroundColor,
              title: Text(
                'Boulder AI Ergebnisse',
                style: TextStyle(color: textColor),
              ),
            ),
            body: Center(
              child: Text(
                'Fehler beim Dekodieren der Bilder: $e',
                style: TextStyle(color: textColor),
              ),
            ),
          );
        }

        final List<dynamic> detections =
            widget.processedResult["detections"] ?? [];
        List<List<Grip>> detectedRoutes = groupGripsIntoRoutes(
          detections,
          distanceThreshold,
        );
        final int originalWidth = widget.processedResult["image_width"];
        final int originalHeight = widget.processedResult["image_height"];

        Map<String, int> detectionToRoute = {};
        for (
          int routeIndex = 0;
          routeIndex < detectedRoutes.length;
          routeIndex++
        ) {
          for (var grip in detectedRoutes[routeIndex]) {
            detectionToRoute[grip.detection["bbox"].toString()] = routeIndex;
          }
        }

        final List<dynamic> detectionsClass =
            widget.processedResult["detections"] ?? [];

        final Map<int, _ClassStats> stats = {};
        for (var det in detectionsClass) {
          final int clsId = det["class"] ?? 0;
          final double conf = (det["confidence"] ?? 0.0).toDouble();
          stats.putIfAbsent(clsId, () => _ClassStats(count: 0, sumConf: 0.0));
          stats[clsId]!.count++;
          stats[clsId]!.sumConf += conf;
        }

        final classEntries =
            stats.entries.toList()..sort((a, b) => a.key.compareTo(b.key));

        double globalSum = 0.0;
        for (var d in detections) {
          globalSum += (d["confidence"] ?? 0.0).toDouble();
        }
        double globalAvg =
            detectionsClass.isEmpty
                ? 0.0
                : (globalSum / detectionsClass.length);

        return Scaffold(
          backgroundColor: backgroundColor,
          appBar: AppBar(
            backgroundColor: backgroundColor,
            title: Text(
              'Boulder AI Ergebnisse',
              style: TextStyle(color: textColor),
            ),
            iconTheme: IconThemeData(color: textColor),
            actions: const [],
          ),
          body: Row(
            children: [
              Expanded(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 20,
                    vertical: 24,
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Center(
                        child: Text(
                          'Analyse abgeschlossen!',
                          style: TextStyle(
                            color: textColor,
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                      const SizedBox(height: 8),
                      Center(
                        child: Text(
                          'Hier sind deine Bilder und die erkannten Griffe:',
                          style: TextStyle(
                            color: textColor.withValues(alpha: 0.8),
                            fontSize: 14,
                          ),
                        ),
                      ),
                      const SizedBox(height: 24),
                      Center(
                        child: Wrap(
                          spacing: 16,
                          runSpacing: 16,
                          alignment: WrapAlignment.center,
                          children: [
                            GestureDetector(
                              onTap: () {
                                _showImageGallery(
                                  context,
                                  [origBytes, detectBytes],
                                  0,
                                  backgroundColor,
                                  textColor,
                                );
                              },
                              child: _buildImageCard(
                                label: 'Original',
                                imageBytes: origBytes,
                                cardColor: cardColor,
                                textColor: textColor,
                              ),
                            ),
                            Container(
                              decoration: BoxDecoration(
                                color: cardColor,
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: Column(
                                children: [
                                  Padding(
                                    padding: const EdgeInsets.all(8.0),
                                    child: Text(
                                      'KI-Detektion',
                                      style: TextStyle(
                                        color: textColor,
                                        fontSize: 16,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ),
                                  SizedBox(
                                    width: 400,
                                    child: AspectRatio(
                                      aspectRatio:
                                          originalWidth / originalHeight,
                                      child: DetectionOverlayInteractive(
                                        imageBytes: origBytes,
                                        detections: detections,
                                        originalWidth: originalWidth.toDouble(),
                                        originalHeight:
                                            originalHeight.toDouble(),
                                        classColorMap: classColorMap,
                                        detectionToRoute: detectionToRoute,
                                        selectedRouteIndex: selectedRouteIndex,
                                        onRouteSelected: (int routeIndex) {
                                          setState(() {
                                            if (selectedRouteIndex ==
                                                routeIndex) {
                                              selectedRouteIndex = null;
                                            } else {
                                              selectedRouteIndex = routeIndex;
                                            }
                                          });
                                        },
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 8),
                      Center(
                        child: Text(
                          'Durchschnittliche Confidence (alle): ${globalAvg.toStringAsFixed(2)}',
                          style: TextStyle(
                            color: textColor.withValues(alpha: 0.9),
                            fontSize: 14,
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: cardColor,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child:
                            classEntries.isEmpty
                                ? Text(
                                  'Keine Griffe erkannt.',
                                  style: TextStyle(color: textColor),
                                )
                                : Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children:
                                      classEntries.map((entry) {
                                        final clsId = entry.key;
                                        final cStats = entry.value;
                                        final avgConf =
                                            cStats.sumConf / cStats.count;

                                        final String name =
                                            (clsId < classNames.length)
                                                ? classNames[clsId]
                                                : 'Unbekannt #$clsId';
                                        final Color labelColor =
                                            classColorMap[clsId] ??
                                            Colors.white70;

                                        return Padding(
                                          padding: const EdgeInsets.symmetric(
                                            vertical: 4.0,
                                          ),
                                          child: Row(
                                            children: [
                                              Container(
                                                width: 16,
                                                height: 16,
                                                margin: const EdgeInsets.only(
                                                  right: 8,
                                                ),
                                                decoration: BoxDecoration(
                                                  color: labelColor,
                                                  borderRadius:
                                                      BorderRadius.circular(4),
                                                ),
                                              ),
                                              Expanded(
                                                child: Text(
                                                  '$name: ${cStats.count}x   Ø Conf: ${avgConf.toStringAsFixed(2)}',
                                                  style: TextStyle(
                                                    color: textColor,
                                                    fontSize: 14,
                                                  ),
                                                ),
                                              ),
                                            ],
                                          ),
                                        );
                                      }).toList(),
                                ),
                      ),
                    ],
                  ),
                ),
              ),
              AnimatedContainer(
                duration: const Duration(milliseconds: 300),
                width: _isRoutesPanelOpen ? 300 : 40,
                decoration: BoxDecoration(
                  color: cardColor,
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(12),
                    bottomLeft: Radius.circular(12),
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withValues(alpha: 0.3),
                      offset: const Offset(-3, 0),
                      blurRadius: 6,
                    ),
                  ],
                ),
                child: SafeArea(
                  child: Stack(
                    children: [
                      if (_isRoutesPanelOpen)
                        Padding(
                          padding: const EdgeInsets.only(
                            left: 40.0,
                            right: 8.0,
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              const SizedBox(height: 12),
                              Text(
                                'Routen',
                                style: TextStyle(
                                  color: textColor,
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Expanded(
                                child: SingleChildScrollView(
                                  child:
                                      detectedRoutes.isEmpty
                                          ? Text(
                                            'Keine Route erkannt.',
                                            style: TextStyle(color: textColor),
                                          )
                                          : Column(
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children:
                                                detectedRoutes.asMap().entries.map((
                                                  entry,
                                                ) {
                                                  int routeIndex = entry.key;
                                                  List<Grip> route =
                                                      entry.value;
                                                  route.sort(
                                                    (a, b) => a.centerY
                                                        .compareTo(b.centerY),
                                                  );
                                                  final routeColor =
                                                      classColorMap[route
                                                          .first
                                                          .classId] ??
                                                      textColor;

                                                  return GestureDetector(
                                                    onTap: () {
                                                      setState(() {
                                                        if (selectedRouteIndex ==
                                                            routeIndex) {
                                                          selectedRouteIndex =
                                                              null;
                                                        } else {
                                                          selectedRouteIndex =
                                                              routeIndex;
                                                        }
                                                      });
                                                    },
                                                    child: Padding(
                                                      padding:
                                                          const EdgeInsets.symmetric(
                                                            vertical: 4.0,
                                                          ),
                                                      child: Row(
                                                        children: [
                                                          Container(
                                                            width: 16,
                                                            height: 16,
                                                            margin:
                                                                const EdgeInsets.only(
                                                                  right: 8,
                                                                ),
                                                            decoration:
                                                                BoxDecoration(
                                                                  color:
                                                                      routeColor,
                                                                  borderRadius:
                                                                      BorderRadius.circular(
                                                                        4,
                                                                      ),
                                                                ),
                                                          ),
                                                          Expanded(
                                                            child: Text(
                                                              'Route ${routeIndex + 1}: (${route.length} Griffe)',
                                                              style: TextStyle(
                                                                color:
                                                                    selectedRouteIndex ==
                                                                            routeIndex
                                                                        ? routeColor
                                                                        : textColor,
                                                                fontSize: 14,
                                                                fontWeight:
                                                                    selectedRouteIndex ==
                                                                            routeIndex
                                                                        ? FontWeight
                                                                            .bold
                                                                        : FontWeight
                                                                            .normal,
                                                              ),
                                                            ),
                                                          ),
                                                        ],
                                                      ),
                                                    ),
                                                  );
                                                }).toList(),
                                          ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      Positioned(
                        left: 0,
                        top: 0,
                        bottom: 0,
                        child: Center(
                          child: IconButton(
                            icon: Icon(
                              _isRoutesPanelOpen
                                  ? Icons.arrow_forward
                                  : Icons.arrow_back,
                              color: textColor,
                            ),
                            onPressed: () {
                              setState(() {
                                _isRoutesPanelOpen = !_isRoutesPanelOpen;
                              });
                            },
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  void _showImageGallery(
    BuildContext context,
    List<Uint8List> images,
    int initialIndex,
    Color backgroundColor,
    Color textColor,
  ) {
    Navigator.push(
      context,
      PageRouteBuilder(
        pageBuilder:
            (context, animation, secondaryAnimation) => ImageGalleryScreen(
              images: images,
              initialIndex: initialIndex,
              backgroundColor: backgroundColor,
              textColor: textColor,
            ),
        transitionsBuilder: (context, animation, secondaryAnimation, child) {
          const begin = Offset(1.0, 0.0);
          const end = Offset.zero;
          const curve = Curves.ease;
          var tween = Tween(
            begin: begin,
            end: end,
          ).chain(CurveTween(curve: curve));
          return SlideTransition(
            position: animation.drive(tween),
            child: child,
          );
        },
        transitionDuration: const Duration(milliseconds: 300),
      ),
    );
  }

  Widget _buildImageCard({
    required String label,
    required Uint8List imageBytes,
    required Color cardColor,
    required Color textColor,
  }) {
    return ConstrainedBox(
      constraints: const BoxConstraints(maxWidth: 400),
      child: Container(
        decoration: BoxDecoration(
          color: cardColor,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Text(
                label,
                style: TextStyle(
                  color: textColor,
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            ClipRRect(
              borderRadius: const BorderRadius.only(
                bottomLeft: Radius.circular(12),
                bottomRight: Radius.circular(12),
              ),
              child: Image.memory(imageBytes, fit: BoxFit.cover),
            ),
          ],
        ),
      ),
    );
  }
}

class _ClassStats {
  int count;
  double sumConf;
  _ClassStats({required this.count, required this.sumConf});
}
