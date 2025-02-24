import 'dart:convert';
import 'dart:typed_data';
import 'package:boulder_ai/util/image_gallery.dart';
import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  final Map<String, dynamic> processedResult;

  const ResultScreen({super.key, required this.processedResult});

  @override
  Widget build(BuildContext context) {
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

              Row(
                mainAxisAlignment: MainAxisAlignment.center, // <-- Hinzugefügt
                crossAxisAlignment: CrossAxisAlignment.start,
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
                  const SizedBox(width: 16),
                  GestureDetector(
                    onTap: () {
                      _showImageGallery(
                        context,
                        [origBytes, detectBytes],
                        1,
                        backgroundColor,
                        textColor,
                      );
                    },
                    child: _buildImageCard(
                      label: 'KI-Detektion',
                      imageBytes: detectBytes,
                      cardColor: cardColor,
                      textColor: textColor,
                    ),
                  ),
                ],
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
    return Container(
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
          ConstrainedBox(
            constraints: const BoxConstraints(
              maxHeight: 800,
            ),
            child: ClipRRect(
              borderRadius: const BorderRadius.only(
                bottomLeft: Radius.circular(12),
                bottomRight: Radius.circular(12),
              ),
              child: Image.memory(
                imageBytes,
                fit: BoxFit.contain,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ClassStats {
  int count;
  double sumConf;
  _ClassStats({required this.count, required this.sumConf});
}
