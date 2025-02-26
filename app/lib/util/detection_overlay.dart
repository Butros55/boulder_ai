import 'dart:typed_data';
import 'package:flutter/material.dart';

class DetectionOverlayInteractive extends StatelessWidget {
  final Uint8List imageBytes;
  final List<dynamic> detections;
  final double originalWidth;
  final double originalHeight;
  final Map<int, Color> classColorMap;
  final Map<String, int> detectionToRoute;
  final int? selectedRouteIndex;
  final ValueChanged<int> onRouteSelected;

  const DetectionOverlayInteractive({
    super.key,
    required this.imageBytes,
    required this.detections,
    required this.originalWidth,
    required this.originalHeight,
    required this.classColorMap,
    required this.detectionToRoute,
    required this.selectedRouteIndex,
    required this.onRouteSelected,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final double scaleX = constraints.maxWidth / originalWidth;
        final double scaleY = constraints.maxHeight / originalHeight;

        return Stack(
          children: [
            Image.memory(
              imageBytes,
              width: constraints.maxWidth,
              height: constraints.maxHeight,
              fit: BoxFit.cover,
            ),
            for (var det in detections) ...{_buildBox(det, scaleX, scaleY)},
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

    final int? routeIndex = detectionToRoute[det["bbox"].toString()];

    final double alpha =
        (selectedRouteIndex != null && routeIndex != selectedRouteIndex)
            ? 0.3
            : 1.0;
    final double borderWidth =
        (selectedRouteIndex != null && routeIndex == selectedRouteIndex)
            ? 3
            : 2;

    return Positioned(
      left: left,
      top: top,
      width: width,
      height: height,
      child: GestureDetector(
        onTap: () {
          if (routeIndex != null) {
            onRouteSelected(routeIndex);
          }
        },
        child: Opacity(
          opacity: alpha,
          child: Container(
            decoration: BoxDecoration(
              border: Border.all(color: color, width: borderWidth),
            ),
          ),
        ),
      ),
    );
  }
}
