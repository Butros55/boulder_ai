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

  @override
  Widget build(BuildContext context) {
    final storage = const FlutterSecureStorage();

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
        final Color accentColor = const Color(0xFF7F5AF0);
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
        try {
          final String origBase64 =
              widget.processedResult["original_image"] as String;
          origBytes = base64Decode(origBase64);
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
        final int originalWidth = widget.processedResult["image_width"];
        final int originalHeight = widget.processedResult["image_height"];

        final List<dynamic> backendRoutes;
        if (widget.processedResult["routes"] is String) {
          backendRoutes =
              jsonDecode(widget.processedResult["routes"] as String)
                  as List<dynamic>;
        } else {
          backendRoutes =
              widget.processedResult["routes"] as List<dynamic>? ?? [];
        }

        Map<String, int> detectionToRoute = {};
        backendRoutes.asMap().forEach((index, routeData) {
          int routeIndex = index; // hier entspricht der Index dem Route-Index
          for (var gripJson in routeData as List<dynamic>) {
            detectionToRoute[gripJson["bbox"].toString()] = routeIndex;
          }
        });

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
          ),
          body: LayoutBuilder(
            builder: (context, constraints) {
              final availableWidth = constraints.maxWidth;

              return Row(
                children: [
                  SizedBox(
                    width: availableWidth,
                    child: Column(
                      children: [
                        Expanded(
                          child: SingleChildScrollView(
                            padding: const EdgeInsets.fromLTRB(20, 0, 20, 0),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(
                                      Icons.check_circle_outline,
                                      color: accentColor,
                                      size: 28,
                                    ),
                                    const SizedBox(width: 8),
                                    Text(
                                      'Analyse abgeschlossen!',
                                      style: TextStyle(
                                        color: textColor,
                                        fontSize: 22,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ],
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
                                    alignment: WrapAlignment.center,
                                    crossAxisAlignment:
                                        WrapCrossAlignment.center,
                                    spacing: 16,
                                    runSpacing: 16,
                                    children: [
                                      GestureDetector(
                                        onTap: () {
                                          _showImageGallery(
                                            context,
                                            [origBytes],
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
                                      Card(
                                        color: cardColor,
                                        shape: RoundedRectangleBorder(
                                          borderRadius: BorderRadius.circular(
                                            12,
                                          ),
                                        ),
                                        child: Column(
                                          children: [
                                            const SizedBox(height: 8),
                                            Text(
                                              'KI-Detektion',
                                              style: TextStyle(
                                                color: textColor,
                                                fontSize: 16,
                                                fontWeight: FontWeight.bold,
                                              ),
                                            ),
                                            const SizedBox(height: 8),
                                            SizedBox(
                                              width: 450,
                                              child: AspectRatio(
                                                aspectRatio:
                                                    originalWidth /
                                                    originalHeight,
                                                child: DetectionOverlayInteractive(
                                                  imageBytes: origBytes,
                                                  detections: detections,
                                                  originalWidth:
                                                      originalWidth.toDouble(),
                                                  originalHeight:
                                                      originalHeight.toDouble(),
                                                  classColorMap: classColorMap,
                                                  detectionToRoute:
                                                      detectionToRoute,
                                                  selectedRouteIndex:
                                                      selectedRouteIndex,
                                                  onRouteSelected: (
                                                    int routeIndex,
                                                  ) {
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
                                                ),
                                              ),
                                            ),
                                          ],
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 16.0),
                          child: Column(
                            children: [
                              Card(
                                color: cardColor,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12),
                                ),
                                child: Padding(
                                  padding: const EdgeInsets.all(16.0),
                                  child: Row(
                                    children: [
                                      Icon(
                                        Icons.trending_up,
                                        color: accentColor,
                                        size: 28,
                                      ),
                                      const SizedBox(width: 12),
                                      Expanded(
                                        child: Text(
                                          'Durchschnittliche Confidence (alle): '
                                          '${globalAvg.toStringAsFixed(2)}',
                                          style: TextStyle(
                                            color: textColor.withValues(
                                              alpha: 0.9,
                                            ),
                                            fontSize: 16,
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                              const SizedBox(height: 8),
                              Card(
                                color: cardColor,
                                shape: const RoundedRectangleBorder(
                                  borderRadius: BorderRadius.vertical(
                                    top: Radius.circular(12),
                                  ),
                                ),
                                child: Padding(
                                  padding: const EdgeInsets.all(16.0),
                                  child:
                                      classEntries.isEmpty
                                          ? Text(
                                            'Keine Griffe erkannt.',
                                            style: TextStyle(color: textColor),
                                          )
                                          : Row(
                                            children: [
                                              Wrap(
                                                children: [
                                                  const SizedBox(width: 32),
                                                  Column(
                                                    crossAxisAlignment:
                                                        CrossAxisAlignment
                                                            .start,
                                                    children: [
                                                      Padding(
                                                        padding:
                                                            const EdgeInsets.only(
                                                              top: 4.0,
                                                            ),
                                                        child: Row(
                                                          children: [
                                                            Icon(
                                                              Icons.route,
                                                              color:
                                                                  accentColor,
                                                              size: 22,
                                                            ),
                                                            const SizedBox(
                                                              width: 8,
                                                            ),
                                                            Text(
                                                              'Routen',
                                                              style: TextStyle(
                                                                color:
                                                                    textColor,
                                                                fontSize: 16,
                                                                fontWeight:
                                                                    FontWeight
                                                                        .bold,
                                                              ),
                                                            ),
                                                          ],
                                                        ),
                                                      ),
                                                      const SizedBox(height: 8),
                                                      backendRoutes.isEmpty
                                                          ? Text(
                                                            'Keine Route erkannt.',
                                                            style: TextStyle(
                                                              color: textColor,
                                                            ),
                                                          )
                                                          : Column(
                                                            crossAxisAlignment:
                                                                CrossAxisAlignment
                                                                    .start,
                                                            children:
                                                                backendRoutes.asMap().entries.map((
                                                                  entry,
                                                                ) {
                                                                  int
                                                                  routeIndex =
                                                                      entry
                                                                          .key; // Der Index in der Liste entspricht dem Route-Index
                                                                  // entry.value sollte eine Liste von Detection-Objekten sein
                                                                  List<Grip>
                                                                  route =
                                                                      (entry.value
                                                                              as List<
                                                                                dynamic
                                                                              >)
                                                                          .map(
                                                                            (
                                                                              e,
                                                                            ) => Grip.fromJson(
                                                                              e
                                                                                  as Map<
                                                                                    String,
                                                                                    dynamic
                                                                                  >,
                                                                            ),
                                                                          )
                                                                          .toList();
                                                                  route.sort(
                                                                    (a, b) => a
                                                                        .centerY
                                                                        .compareTo(
                                                                          b.centerY,
                                                                        ),
                                                                  );
                                                                  final routeColor =
                                                                      classColorMap[route
                                                                          .first
                                                                          .classId] ??
                                                                      textColor;
                                                                  return GestureDetector(
                                                                    onTap: () {
                                                                      setState(() {
                                                                        selectedRouteIndex =
                                                                            (selectedRouteIndex ==
                                                                                    routeIndex)
                                                                                ? null
                                                                                : routeIndex;
                                                                      });
                                                                    },
                                                                    child: Padding(
                                                                      padding: const EdgeInsets.symmetric(
                                                                        vertical:
                                                                            4.0,
                                                                      ),
                                                                      child: Row(
                                                                        children: [
                                                                          Container(
                                                                            width:
                                                                                16,
                                                                            height:
                                                                                16,
                                                                            margin: const EdgeInsets.only(
                                                                              right:
                                                                                  8,
                                                                            ),
                                                                            decoration: BoxDecoration(
                                                                              color:
                                                                                  routeColor,
                                                                              borderRadius: BorderRadius.circular(
                                                                                4,
                                                                              ),
                                                                            ),
                                                                          ),
                                                                          Text(
                                                                            'Route ${routeIndex + 1}: (${route.length} Griffe)',
                                                                            style: TextStyle(
                                                                              color:
                                                                                  selectedRouteIndex ==
                                                                                          routeIndex
                                                                                      ? routeColor
                                                                                      : textColor,
                                                                              fontSize:
                                                                                  14,
                                                                              fontWeight:
                                                                                  selectedRouteIndex ==
                                                                                          routeIndex
                                                                                      ? FontWeight.bold
                                                                                      : FontWeight.normal,
                                                                            ),
                                                                          ),
                                                                        ],
                                                                      ),
                                                                    ),
                                                                  );
                                                                }).toList(),
                                                          ),
                                                    ],
                                                  ),
                                                  const SizedBox(width: 32),

                                                  Column(
                                                    crossAxisAlignment:
                                                        CrossAxisAlignment
                                                            .start,
                                                    children: [
                                                      Row(
                                                        children: [
                                                          Icon(
                                                            Icons.list,
                                                            color: accentColor,
                                                            size: 28,
                                                          ),
                                                          const SizedBox(
                                                            width: 8,
                                                          ),
                                                          Text(
                                                            'Erkannte Griffe',
                                                            style: TextStyle(
                                                              color: textColor,
                                                              fontSize: 16,
                                                              fontWeight:
                                                                  FontWeight
                                                                      .bold,
                                                            ),
                                                          ),
                                                        ],
                                                      ),
                                                      const SizedBox(height: 8),
                                                      ...classEntries.map((
                                                        entry,
                                                      ) {
                                                        final clsId = entry.key;
                                                        final cStats =
                                                            entry.value;
                                                        final avgConf =
                                                            cStats.sumConf /
                                                            cStats.count;
                                                        final String name =
                                                            (clsId <
                                                                    classNames
                                                                        .length)
                                                                ? classNames[clsId]
                                                                : 'Unbekannt #$clsId';
                                                        final Color labelColor =
                                                            classColorMap[clsId] ??
                                                            Colors.white70;

                                                        return Padding(
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
                                                                decoration: BoxDecoration(
                                                                  color:
                                                                      labelColor,
                                                                  borderRadius:
                                                                      BorderRadius.circular(
                                                                        4,
                                                                      ),
                                                                ),
                                                              ),
                                                              Text(
                                                                '$name: ${cStats.count}x   Ø Conf: ${avgConf.toStringAsFixed(2)}',
                                                                style: TextStyle(
                                                                  color:
                                                                      textColor,
                                                                  fontSize: 14,
                                                                ),
                                                              ),
                                                            ],
                                                          ),
                                                        );
                                                      }),
                                                    ],
                                                  ),
                                                ],
                                              ),
                                            ],
                                          ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              );
            },
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
      ),
    );
  }

  Widget _buildImageCard({
    required String label,
    required Uint8List imageBytes,
    required Color cardColor,
    required Color textColor,
  }) {
    return Card(
      color: cardColor,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      elevation: 2,
      child: Column(
        children: [
          const SizedBox(height: 8),
          Text(
            label,
            style: TextStyle(
              color: textColor,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: const BorderRadius.only(
              bottomLeft: Radius.circular(12),
              bottomRight: Radius.circular(12),
            ),
            child: Image.memory(imageBytes, fit: BoxFit.cover, width: 450),
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
