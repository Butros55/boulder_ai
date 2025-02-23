import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:photo_view/photo_view.dart';

class ResultScreen extends StatelessWidget {
  // Erwartet eine Map, die "original_image", "filtered_image", "detection_image" und "detections" enthält.
  final Map<String, dynamic> processedResult;

  const ResultScreen({super.key, required this.processedResult});

  @override
  Widget build(BuildContext context) {
    // Dekodiere die drei Bilder
    Uint8List origBytes;
    Uint8List filtBytes;
    Uint8List detectBytes;

    try {
      final String origBase64 = processedResult["original_image"] as String;
      final String filtBase64 = processedResult["filtered_image"] as String;
      final String detectBase64 = processedResult["detection_image"] as String;

      origBytes = base64Decode(origBase64);
      filtBytes = base64Decode(filtBase64);
      detectBytes = base64Decode(detectBase64);
    } catch (e) {
      return Scaffold(
        appBar: AppBar(title: const Text('Ergebnisse')),
        body: Center(child: Text('Fehler beim Dekodieren der Bilder: $e')),
      );
    }

    String jsonString = const JsonEncoder.withIndent(
      '  ',
    ).convert(processedResult["detections"]);

    return Scaffold(
      appBar: AppBar(title: const Text('Boulder AI Ergebnisse')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _buildImageContainer(context, origBytes, 'Original'),
                const SizedBox(width: 16),
                _buildImageContainer(context, detectBytes, 'KI-Detektion'),
              ],
            ),
            const SizedBox(height: 20),
            Text(
              'Erkannte Griff-Daten:',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            const SizedBox(height: 10),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12.0),
              decoration: BoxDecoration(
                color: Colors.grey.shade200,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                jsonString,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 14),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // Hilfsfunktion, die ein Bild in einem Container anzeigt und beim Tippen vergrößert
  Widget _buildImageContainer(
    BuildContext context,
    Uint8List bytes,
    String label,
  ) {
    return GestureDetector(
      onTap: () {
        showDialog(
          context: context,
          builder: (BuildContext context) {
            return Dialog(
              insetPadding: const EdgeInsets.all(16.0),
              child: SizedBox(
                width: double.infinity,
                height: double.infinity,
                child: PhotoView(
                  imageProvider: MemoryImage(bytes),
                  minScale: PhotoViewComputedScale.contained,
                  maxScale: PhotoViewComputedScale.covered * 3.0,
                  backgroundDecoration: BoxDecoration(color: Colors.white),
                ),
              ),
            );
          },
        );
      },
      child: Column(
        children: [
          Container(
            width: 450,
            height: 750,
            decoration: BoxDecoration(
              border: Border.all(color: Colors.grey, width: 2),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Image.memory(bytes, fit: BoxFit.contain),
          ),
          const SizedBox(height: 8),
          Text(label),
        ],
      ),
    );
  }
}
