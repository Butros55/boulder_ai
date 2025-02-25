import 'dart:convert';
import 'dart:typed_data';
import 'package:boulder_ai/util/image_gallery.dart';
import 'package:boulder_ai/util/util.dart';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class ResultScreen extends StatelessWidget {
  final Map<String, dynamic> processedResult;

  const ResultScreen({super.key, required this.processedResult});

  @override
  Widget build(BuildContext context) {
    final storage = const FlutterSecureStorage();
    const double distanceThreshold = 320.0;

    // FutureBuilder, der prüft, ob ein gültiges Token existiert.
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
          // Token fehlt: SnackBar anzeigen und zum Login navigieren.
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

        // Falls Token vorhanden, fahre mit der Anzeige fort.
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

        Uint8List origBytes;
        Uint8List detectBytes;
        try {
          final String origBase64 = processedResult["original_image"] as String;
          final String detectBase64 = processedResult["detection_image"] as String;

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

        final List<dynamic> detections = processedResult["detections"] ?? [];
        List<List<Grip>> detectedRoutes = groupGripsIntoRoutes(detections, distanceThreshold);
        final int originalWidth = processedResult["image_width"];
        final int originalHeight = processedResult["image_height"];



        final Map<int, _ClassStats> stats = {};
        for (var det in detections) {
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
            detections.isEmpty ? 0.0 : (globalSum / detections.length);

        return Scaffold(
          backgroundColor: backgroundColor,
          appBar: AppBar(
            backgroundColor: backgroundColor,
            title: Text(
              'Boulder AI Ergebnisse',
              style: TextStyle(color: textColor),
            ),
            iconTheme: IconThemeData(color: textColor),
          ),
          body: SafeArea(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
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
                        // Hier der ersetzte KI-Detektion-Bereich mit DetectionOverlay:
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
                                  child: DetectionOverlay(
                                    imageBytes: origBytes, // oder detectBytes, falls du es bevorzugst
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
                                      8: const Color(0xFF8D6E63), // wood
                                      9: Colors.yellow,
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

                  const SizedBox(height: 30),
                  Center(
                    child: Text(
                      'Zusammenfassung der erkannten Griffe',
                      style: TextStyle(
                        color: textColor,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  const SizedBox(height: 30),
                  // Neue Sektion: Erkannte Routen
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
                              // Hier sortieren wir die Griffe (zum Beispiel von unten nach oben)
                              route.sort((a, b) => a.centerY.compareTo(b.centerY));
                              // Erzeuge einen String aus den Klassen-IDs der Route
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
                    child: classEntries.isEmpty
                        ? Text(
                            'Keine Griffe erkannt.',
                            style: TextStyle(color: textColor),
                          )
                        : Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: classEntries.map((entry) {
                              final clsId = entry.key;
                              final cStats = entry.value;
                              final avgConf = cStats.sumConf / cStats.count;

                              final String name = (clsId < classNames.length)
                                  ? classNames[clsId]
                                  : 'Unbekannt #$clsId';
                              final Color labelColor =
                                  classColorMap[name] ?? Colors.white70;

                              return Padding(
                                padding: const EdgeInsets.symmetric(vertical: 4.0),
                                child: Row(
                                  children: [
                                    Container(
                                      width: 16,
                                      height: 16,
                                      margin: const EdgeInsets.only(right: 8),
                                      decoration: BoxDecoration(
                                        color: labelColor,
                                        borderRadius: BorderRadius.circular(4),
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
      constraints: const BoxConstraints(
        maxWidth: 500,
      ),
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


class DetectionOverlay extends StatelessWidget {
  final Uint8List imageBytes;
  final List<dynamic> detections;
  final double originalWidth;
  final double originalHeight;

  // classColorMap: falls du pro Klasse eine andere Farbe willst
  final Map<int, Color> classColorMap;

  const DetectionOverlay({
    Key? key,
    required this.imageBytes,
    required this.detections,
    required this.originalWidth,
    required this.originalHeight,
    required this.classColorMap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        // So skalierst du die Koordinaten auf die aktuelle Widget-Größe
        final double scaleX = constraints.maxWidth / originalWidth;
        final double scaleY = constraints.maxHeight / originalHeight;

        return Stack(
          children: [
            // 1) Bild als Hintergrund
            Image.memory(
              imageBytes,
              width: constraints.maxWidth,
              height: constraints.maxHeight,
              fit: BoxFit.cover,
            ),

            // 2) Jede Detection als Positioned
            for (var det in detections) ...{
              _buildBox(det, scaleX, scaleY),
            },
          ],
        );
      },
    );
  }

  Widget _buildBox(dynamic det, double scaleX, double scaleY) {
    final bbox = det["bbox"] as List<dynamic>;
    final double x1 = bbox[0];
    final double y1 = bbox[1];
    final double x2 = bbox[2];
    final double y2 = bbox[3];

    final double left = x1 * scaleX;
    final double top = y1 * scaleY;
    final double width = (x2 - x1) * scaleX;
    final double height = (y2 - y1) * scaleY;

    final int clsId = det["class"];
    final Color color = classColorMap[clsId] ?? Colors.white;

    return Positioned(
      left: left,
      top: top,
      width: width,
      height: height,
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(color: color, width: 2),
        ),
      ),
    );
  }
}
