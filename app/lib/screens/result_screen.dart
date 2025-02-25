import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'package:boulder_ai/util/detection_overlay.dart';
import 'package:boulder_ai/util/image_gallery.dart';
import 'package:boulder_ai/util/util.dart';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

/// Das Grip-Modell (wie vorher) berechnet den Mittelpunkt einer Detection.
class Grip {
  final int classId;
  final double centerX;
  final double centerY;
  final Map detection; // enthält bspw. bbox, confidence etc.

  Grip({
    required this.classId,
    required this.centerX,
    required this.centerY,
    required this.detection,
  });
}

/// Gruppiert Detections in Routen mittels DFS, wie zuvor.
List<List<Grip>> groupGripsIntoRoutes(List<dynamic> detections, double threshold) {
  List<Grip> allGrips = [];
  for (var d in detections) {
    final List<dynamic> bbox = d["bbox"];
    double x1 = bbox[0].toDouble();
    double y1 = bbox[1].toDouble();
    double x2 = bbox[2].toDouble();
    double y2 = bbox[3].toDouble();
    double centerX = (x1 + x2) / 2;
    double centerY = (y1 + y2) / 2;
    int classId = d["class"] ?? 0;
    allGrips.add(Grip(
      classId: classId,
      centerX: centerX,
      centerY: centerY,
      detection: d,
    ));
  }
  // Gruppiere nach Farbe (classId)
  Map<int, List<Grip>> gripsByColor = {};
  for (var grip in allGrips) {
    gripsByColor.putIfAbsent(grip.classId, () => []).add(grip);
  }
  List<List<Grip>> routes = [];
  for (var group in gripsByColor.values) {
    routes.addAll(_findConnectedComponents(group, threshold));
  }
  return routes;
}

List<List<Grip>> _findConnectedComponents(List<Grip> grips, double threshold) {
  List<List<Grip>> components = [];
  Set<int> visited = {};
  for (int i = 0; i < grips.length; i++) {
    if (!visited.contains(i)) {
      List<Grip> component = [];
      _dfs(i, grips, threshold, visited, component);
      components.add(component);
    }
  }
  return components;
}

void _dfs(int index, List<Grip> grips, double threshold, Set<int> visited, List<Grip> component) {
  visited.add(index);
  component.add(grips[index]);
  for (int j = 0; j < grips.length; j++) {
    if (!visited.contains(j)) {
      double dx = grips[index].centerX - grips[j].centerX;
      double dy = grips[index].centerY - grips[j].centerY;
      double distance = sqrt(dx * dx + dy * dy);
      if (distance < threshold) {
        _dfs(j, grips, threshold, visited, component);
      }
    }
  }
}

/// INTERAKTIVER ResultScreen, der den aktuell selektierten Routenindex speichert.
class ResultScreen extends StatefulWidget {
  final Map<String, dynamic> processedResult;

  const ResultScreen({Key? key, required this.processedResult}) : super(key: key);

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  int? selectedRouteIndex; // aktuell selektierte Route

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

        // Basisfarben und Styles
        final Color backgroundColor = const Color(0xFF1C1C1E);
        final Color cardColor = const Color(0xFF2C2C2E);
        final Color textColor = Colors.white;

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

        final Map<String, Color> classColorMap = {
          "black": Colors.grey.shade800,
          "blue": Colors.blueAccent,
          "grey": Colors.blueGrey,
          "orange": Colors.deepOrangeAccent,
          "purple": Colors.deepPurpleAccent,
          "red": Colors.redAccent,
          "turquoise": Colors.cyanAccent,
          "white": Colors.white70,
          "wood": const Color(0xFF8D6E63),
          "yellow": Colors.amberAccent,
        };

        // Bilder decodieren
        Uint8List origBytes;
        Uint8List detectBytes;
        try {
          final String origBase64 = widget.processedResult["original_image"] as String;
          final String detectBase64 = widget.processedResult["detection_image"] as String;
          origBytes = base64Decode(origBase64);
          detectBytes = base64Decode(detectBase64);
        } catch (e) {
          return Scaffold(
            backgroundColor: backgroundColor,
            appBar: AppBar(
              backgroundColor: backgroundColor,
              title: Text('Boulder AI Ergebnisse', style: TextStyle(color: textColor)),
            ),
            body: Center(
              child: Text('Fehler beim Dekodieren der Bilder: $e', style: TextStyle(color: textColor)),
            ),
          );
        }

        final List<dynamic> detections = widget.processedResult["detections"] ?? [];
        // Berechne die Routen anhand der Detections
        List<List<Grip>> detectedRoutes = groupGripsIntoRoutes(detections, distanceThreshold);
        final int originalWidth = widget.processedResult["image_width"];
        final int originalHeight = widget.processedResult["image_height"];

        // Zusätzlich: Erstelle ein Mapping von jeder Detection (über bbox als String) zur Route
        Map<String, int> detectionToRoute = {};
        for (int routeIndex = 0; routeIndex < detectedRoutes.length; routeIndex++) {
          for (var grip in detectedRoutes[routeIndex]) {
            // Nutze bspw. den String der bbox als Key
            detectionToRoute[grip.detection["bbox"].toString()] = routeIndex;
          }
        }

        // (Weitere Statistiken aus den Detections ...)

        return Scaffold(
          backgroundColor: backgroundColor,
          appBar: AppBar(
            backgroundColor: backgroundColor,
            title: Text('Boulder AI Ergebnisse', style: TextStyle(color: textColor)),
            iconTheme: IconThemeData(color: textColor),
          ),
          body: SafeArea(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Überschrift
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
                        color: textColor.withOpacity(0.8),
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
                        // Originalbild-Karte
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
                        // KI-Detektion als interaktives Overlay
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
                                width: 500,
                                child: AspectRatio(
                                  aspectRatio: originalWidth / originalHeight,
                                  child: DetectionOverlayInteractive(
                                    imageBytes: origBytes,
                                    detections: detections,
                                    originalWidth: originalWidth.toDouble(),
                                    originalHeight: originalHeight.toDouble(),
                                    classColorMap: {
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
                                    },
                                    detectionToRoute: detectionToRoute,
                                    selectedRouteIndex: selectedRouteIndex,
                                    onRouteSelected: (int routeIndex) {
                                      setState(() {
                                        // Toggle-Auswahl: Klicke auf dieselbe Route => Auswahl aufheben
                                        if (selectedRouteIndex == routeIndex) {
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
                  // ... (weitere UI-Komponenten, z. B. Zusammenfassung, Routenliste etc.)
                  const SizedBox(height: 30),
                  Center(
                    child: Text(
                      'Erkannte Routen',
                      style: TextStyle(
                        color: textColor,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: cardColor,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: detectedRoutes.isEmpty
                        ? Text(
                            'Keine Route erkannt.',
                            style: TextStyle(color: textColor),
                          )
                        : Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: detectedRoutes.asMap().entries.map((entry) {
                              int routeIndex = entry.key;
                              List<Grip> route = entry.value;
                              route.sort((a, b) => a.centerY.compareTo(b.centerY));
                              String routeSummary = route.map((grip) => classNames[grip.classId]).join(' → ');
                              return Padding(
                                padding: const EdgeInsets.symmetric(vertical: 4.0),
                                child: Text(
                                  'Route ${routeIndex + 1}: (${route.length} Griffe) - $routeSummary',
                                  style: TextStyle(color: textColor, fontSize: 14),
                                ),
                              );
                            }).toList(),
                          ),
                  ),
                  // ... weitere Abschnitte
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  void _showImageGallery(BuildContext context, List<Uint8List> images, int initialIndex, Color backgroundColor, Color textColor) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ImageGalleryScreen(
          images: images,
          initialIndex: initialIndex,
          backgroundColor: backgroundColor,
          textColor: textColor,
        ),
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
      constraints: const BoxConstraints(maxWidth: 500),
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
              child: Image.memory(
                imageBytes,
                fit: BoxFit.cover,
              ),
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
