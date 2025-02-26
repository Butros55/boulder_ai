import 'dart:typed_data';
import 'package:flutter/material.dart';

/// Eine eigenständige Seite für die Vollbild-Galerie.
/// Du kannst sie in beliebigen Screens via Navigator.push() aufrufen.
class ImageGalleryScreen extends StatefulWidget {
  final List<Uint8List> images;
  final int initialIndex;
  final Color backgroundColor;
  final Color textColor;

  const ImageGalleryScreen({
    super.key,
    required this.images,
    required this.initialIndex,
    required this.backgroundColor,
    required this.textColor,
  });

  @override
  State<ImageGalleryScreen> createState() => _ImageGalleryScreenState();
}

class _ImageGalleryScreenState extends State<ImageGalleryScreen> {
  late PageController _pageController;

  @override
  void initState() {
    super.initState();
    _pageController = PageController(initialPage: widget.initialIndex);
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: widget.backgroundColor,
      appBar: AppBar(
        backgroundColor: widget.backgroundColor,
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: widget.textColor),
          onPressed: () => Navigator.of(context).pop(),
        ),
        elevation: 0,
        title: Text(
          'Vollbild-Ansicht',
          style: TextStyle(color: widget.textColor),
        ),
      ),
      // PageView, um zwischen den Bildern zu wischen
      body: PageView.builder(
        controller: _pageController,
        itemCount: widget.images.length,
        itemBuilder: (context, index) {
          // InteractiveViewer erlaubt Pinch-to-Zoom
          return InteractiveViewer(
            child: Center(
              child: Image.memory(widget.images[index], fit: BoxFit.contain),
            ),
          );
        },
      ),
    );
  }
}
