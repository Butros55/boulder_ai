import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:boulder_ai/util/detection_overlay.dart';
import 'package:boulder_ai/util/fullscreen_viewer.dart';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

const Color _kBackgroundColor = Color(0xFF1C1C1E);
const Color _kCardColor = Color(0xFF2C2C2E);
const Color _kAccentColor = Color(0xFF7F5AF0);
const Color _kTextColor = Colors.white;

class ResultScreen extends StatefulWidget {
  final Map<String, dynamic> processedResult;

  const ResultScreen({super.key, required this.processedResult});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  late final Future<String?> _tokenFuture = const FlutterSecureStorage().read(key: 'jwt_token');

  int? _selectedRouteIndex;
  bool _showOverlay = true;
  bool _useShapes = true;
  bool _showBeta = false;
  Map? _tappedGrip;

  /// In der Zugfolge-Karte ausgewählter Schritt (Index in der Schrittliste
  /// der gewählten Route) samt zugehörigem Griff fürs Bild-Highlight.
  int? _selectedBetaStepIndex;
  Map? _betaStepGrip;

  String? _decodeError;
  Uint8List? _origBytes;
  List<dynamic> _detections = const [];
  List<dynamic> _routes = const [];
  List<List<BetaStep>?> _betas = const [];
  Map<String, int> _detectionToRoute = {};
  List<_RouteInfo> _routeInfos = const [];
  double _imgW = 800;
  double _imgH = 600;
  bool _hasSegmentation = false;

  int _holdCount = 0;
  int _volumeCount = 0;
  double _avgConfidence = 0;
  List<_ColorStat> _colorStats = const [];

  // Fallback für alte Backend-Stände; wird aus color_hex der API-Antwort überschrieben.
  final Map<int, Color> _classColorMap = {
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
    10: Colors.green,
    11: Colors.pink,
  };

  final Map<int, String> _classNames = {
    0: "black",
    1: "blue",
    2: "grey",
    3: "orange",
    4: "purple",
    5: "red",
    6: "turquoise",
    7: "white",
    8: "wood",
    9: "yellow",
    10: "green",
    11: "pink",
  };

  @override
  void initState() {
    super.initState();
    _parseResult();
  }

  void _parseResult() {
    try {
      _origBytes = base64Decode(widget.processedResult["original_image"] as String);
    } catch (e) {
      _decodeError = 'Fehler beim Dekodieren der Bilder: $e';
      return;
    }

    _detections = widget.processedResult["detections"] as List<dynamic>? ?? const [];

    // Farbe und Name pro Klasse aus der API-Antwort übernehmen, wenn vorhanden.
    for (final det in _detections) {
      if (det is! Map) {
        continue;
      }
      final int clsId = det["class"] is int ? det["class"] as int : 0;
      final dynamic hex = det["color_hex"] ?? (det["color"] is Map ? det["color"]["hex"] : null);
      if (hex is String) {
        final int? parsed = int.tryParse(hex.replaceFirst('#', ''), radix: 16);
        if (parsed != null) {
          _classColorMap[clsId] = Color(0xFF000000 | parsed);
        }
      }
      final dynamic name = det["class_name"] ?? (det["color"] is Map ? det["color"]["name"] : null);
      if (name is String) {
        _classNames[clsId] = name;
      }
    }

    _imgW = ((widget.processedResult["image_width"] as num?) ?? 800).toDouble();
    _imgH = ((widget.processedResult["image_height"] as num?) ?? 600).toDouble();

    final dynamic rawRoutes = widget.processedResult["routes"];
    if (rawRoutes is String) {
      try {
        _routes = jsonDecode(rawRoutes) as List<dynamic>;
      } catch (_) {
        _routes = const [];
      }
    } else {
      _routes = rawRoutes as List<dynamic>? ?? const [];
    }

    // Zugvorschläge (Beta): ältere gespeicherte Analysen haben kein
    // betas-Feld, parseBetas liefert dann nur null-Einträge.
    _betas = parseBetas(widget.processedResult["betas"], _routes.length);

    _detectionToRoute = {};
    final routeInfos = <_RouteInfo>[];
    for (int i = 0; i < _routes.length; i++) {
      final dynamic route = _routes[i];
      if (route is! List) {
        continue;
      }
      for (final grip in route) {
        if (grip is Map && grip["bbox"] != null) {
          _detectionToRoute[grip["bbox"].toString()] = i;
        }
      }
      final Map? firstGrip = route.isNotEmpty && route.first is Map ? route.first as Map : null;
      routeInfos.add(
        _RouteInfo(
          index: i,
          color: firstGrip != null ? resolveDetectionColor(firstGrip, _classColorMap) : _kAccentColor,
          gripCount: route.length,
        ),
      );
    }
    _routeInfos = routeInfos;

    _hasSegmentation = _detections.any(
      (d) => d is Map && d["segmentation"] is List && (d["segmentation"] as List).isNotEmpty,
    );
    _useShapes = _hasSegmentation;

    double sumConf = 0;
    final Map<int, _ColorStat> byClass = {};
    for (final det in _detections) {
      if (det is! Map) {
        continue;
      }
      final double conf = ((det["confidence"] as num?) ?? 0).toDouble();
      sumConf += conf;
      if (det["det_class"] == "volume") {
        _volumeCount++;
      } else {
        _holdCount++;
      }
      final int clsId = det["class"] is int ? det["class"] as int : 0;
      final _ColorStat stat = byClass.putIfAbsent(
        clsId,
        () => _ColorStat(
          name: _capitalize(_classNames[clsId] ?? 'Unbekannt #$clsId'),
          color: _classColorMap[clsId] ?? Colors.white70,
        ),
      );
      stat.count++;
      stat.sumConf += conf;
    }
    _avgConfidence = _detections.isEmpty ? 0 : sumConf / _detections.length;
    _colorStats = byClass.values.toList()..sort((a, b) => b.count.compareTo(a.count));
  }

  /// Zugfolge der gerade ausgewählten Route, sonst null.
  List<BetaStep>? get _selectedBeta {
    final int? sel = _selectedRouteIndex;
    if (sel == null || sel < 0 || sel >= _betas.length) {
      return null;
    }
    return _betas[sel];
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<String?>(
      future: _tokenFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Scaffold(
            backgroundColor: _kBackgroundColor,
            body: Center(child: CircularProgressIndicator()),
          );
        }
        if (snapshot.data == null) {
          WidgetsBinding.instance.addPostFrameCallback((_) {
            if (!mounted) {
              return;
            }
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
          return const Scaffold(
            backgroundColor: _kBackgroundColor,
            body: Center(child: Text("Leite um...", style: TextStyle(color: _kTextColor))),
          );
        }

        if (_decodeError != null || _origBytes == null) {
          return Scaffold(
            backgroundColor: _kBackgroundColor,
            appBar: _buildAppBar(),
            body: Center(
              child: Text(
                _decodeError ?? 'Bilddaten fehlen.',
                style: const TextStyle(color: _kTextColor),
              ),
            ),
          );
        }

        return Scaffold(
          backgroundColor: _kBackgroundColor,
          appBar: _buildAppBar(),
          body: Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
            child: LayoutBuilder(
              builder: (context, constraints) {
                final bool isWide = constraints.maxWidth >= 880;
                if (isWide) {
                  return Row(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Expanded(child: _buildImageCard()),
                      const SizedBox(width: 16),
                      SizedBox(
                        width: 340,
                        child: SingleChildScrollView(child: _buildSidebarContent()),
                      ),
                    ],
                  );
                }
                return SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      SizedBox(
                        height: math.max(380.0, constraints.maxHeight * 0.7),
                        child: _buildImageCard(),
                      ),
                      const SizedBox(height: 16),
                      _buildSidebarContent(),
                    ],
                  ),
                );
              },
            ),
          ),
        );
      },
    );
  }

  AppBar _buildAppBar() {
    return AppBar(
      backgroundColor: _kBackgroundColor,
      title: const Text('Boulder AI Ergebnisse', style: TextStyle(color: _kTextColor)),
      iconTheme: const IconThemeData(color: _kTextColor),
    );
  }

  Widget _buildImageCard() {
    return Card(
      color: _kCardColor,
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(12, 6, 12, 12),
        child: Column(
          children: [
            SizedBox(
              height: 44,
              child: Row(
                children: [
                  const Icon(Icons.auto_awesome, color: _kAccentColor, size: 18),
                  const SizedBox(width: 8),
                  const Expanded(
                    child: Text(
                      'Boulderwand',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(color: _kTextColor, fontSize: 15, fontWeight: FontWeight.bold),
                    ),
                  ),
                  Text(
                    'Overlay',
                    style: TextStyle(color: _kTextColor.withValues(alpha: 0.7), fontSize: 12),
                  ),
                  const SizedBox(width: 2),
                  Switch(
                    value: _showOverlay,
                    onChanged: (value) => setState(() => _showOverlay = value),
                    materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    thumbColor: WidgetStateProperty.resolveWith(
                      (states) =>
                          states.contains(WidgetState.selected) ? _kAccentColor : Colors.white70,
                    ),
                    trackColor: WidgetStateProperty.resolveWith(
                      (states) => states.contains(WidgetState.selected)
                          ? _kAccentColor.withValues(alpha: 0.4)
                          : Colors.white24,
                    ),
                  ),
                  if (_selectedBeta != null)
                    Padding(
                      padding: const EdgeInsets.only(left: 6),
                      child: _BetaToggle(
                        active: _showBeta,
                        onChanged: (value) => setState(() => _showBeta = value),
                      ),
                    ),
                  if (_hasSegmentation)
                    Padding(
                      padding: const EdgeInsets.only(left: 6, right: 2),
                      child: _ShapesToggle(
                        useShapes: _useShapes,
                        onChanged: (value) => setState(() => _useShapes = value),
                      ),
                    ),
                  IconButton(
                    tooltip: 'Vollbild (Zoom mit Mausrad)',
                    visualDensity: VisualDensity.compact,
                    onPressed: _openFullscreen,
                    icon: Icon(Icons.fullscreen, color: _kTextColor.withValues(alpha: 0.7)),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 4),
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: MouseRegion(
                  cursor: SystemMouseCursors.click,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      DetectionOverlayInteractive(
                        imageBytes: _origBytes!,
                        detections: _detections,
                        routes: _routes,
                        betas: _betas,
                        showBeta: _showBeta,
                        originalWidth: _imgW,
                        originalHeight: _imgH,
                        classColorMap: _classColorMap,
                        detectionToRoute: _detectionToRoute,
                        selectedRouteIndex: _selectedRouteIndex,
                        showOverlay: _showOverlay,
                        useShapes: _useShapes,
                        highlightedDetection: _tappedGrip ?? _betaStepGrip,
                        onGripTapped: _handleGripTap,
                        onEmptyTapped: _handleEmptyTap,
                      ),
                      Positioned(
                        right: 8,
                        bottom: 8,
                        child: GestureDetector(
                          onTap: _openFullscreen,
                          child: Container(
                            padding: const EdgeInsets.all(6),
                            decoration: BoxDecoration(
                              color: Colors.black.withValues(alpha: 0.45),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: const Icon(Icons.zoom_in, size: 16, color: Colors.white),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSidebarContent() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _buildStats(),
        if (_routeInfos.isNotEmpty) ...[
          const SizedBox(height: 12),
          _buildRoutesCard(),
        ],
        if (_selectedBeta != null) ...[
          const SizedBox(height: 12),
          _buildBetaCard(_selectedBeta!),
        ],
        if (_tappedGrip != null) ...[
          const SizedBox(height: 12),
          _GripInfoCard(
            det: _tappedGrip!,
            color: resolveDetectionColor(_tappedGrip!, _classColorMap),
            colorName: _detectionColorName(_tappedGrip!, _classNames),
            routeIndex: _detectionToRoute[_tappedGrip!["bbox"].toString()],
            onClose: () => setState(() => _tappedGrip = null),
          ),
        ],
        const SizedBox(height: 12),
        _buildColorDistribution(),
      ],
    );
  }

  Widget _buildStats() {
    return LayoutBuilder(
      builder: (context, constraints) {
        final List<Widget> tiles = [
          _statTile(icon: Icons.fitness_center, value: '$_holdCount', label: 'Griffe'),
          _statTile(icon: Icons.change_history, value: '$_volumeCount', label: 'Volumes'),
          _statTile(icon: Icons.route, value: '${_routeInfos.length}', label: 'Routen'),
          _statTile(
            icon: Icons.speed,
            value: '${(_avgConfidence * 100).toStringAsFixed(0)} %',
            label: 'Ø Confidence',
          ),
        ];
        if (constraints.maxWidth >= 720) {
          return Row(
            children: [
              for (int i = 0; i < tiles.length; i++) ...[
                if (i > 0) const SizedBox(width: 12),
                Expanded(child: tiles[i]),
              ],
            ],
          );
        }
        final double tileWidth = (constraints.maxWidth - 12) / 2;
        return Wrap(
          spacing: 12,
          runSpacing: 12,
          children: [for (final tile in tiles) SizedBox(width: tileWidth, child: tile)],
        );
      },
    );
  }

  Widget _statTile({required IconData icon, required String value, required String label}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: _kCardColor,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: _kAccentColor.withValues(alpha: 0.16),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, color: _kAccentColor, size: 18),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  value,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(color: _kTextColor, fontSize: 17, fontWeight: FontWeight.bold),
                ),
                Text(
                  label,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(color: _kTextColor.withValues(alpha: 0.55), fontSize: 11),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRoutesCard() {
    return Card(
      color: _kCardColor,
      elevation: 2,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.route, color: _kAccentColor, size: 20),
                SizedBox(width: 8),
                Text(
                  'Routen',
                  style: TextStyle(color: _kTextColor, fontSize: 15, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 10),
            _RouteChipBar(
              routes: _routeInfos,
              selectedIndex: _selectedRouteIndex,
              wrap: true,
              onSelected: (index) => setState(() {
                _selectedRouteIndex = index;
                _tappedGrip = null;
                _selectedBetaStepIndex = null;
                _betaStepGrip = null;
              }),
            ),
          ],
        ),
      ),
    );
  }

  /// Karte "Zugfolge": Schrittliste der Zugvorschläge (Beta) für die
  /// ausgewählte Route. Antippen hebt den zugehörigen Griff im Bild hervor.
  Widget _buildBetaCard(List<BetaStep> steps) {
    final int? sel = _selectedRouteIndex;
    final List route =
        sel != null && sel < _routes.length && _routes[sel] is List ? _routes[sel] as List : const [];
    return Card(
      color: _kCardColor,
      elevation: 2,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.moving, color: _kAccentColor, size: 20),
                const SizedBox(width: 8),
                const Text(
                  'Zugfolge',
                  style: TextStyle(color: _kTextColor, fontSize: 15, fontWeight: FontWeight.bold),
                ),
                const SizedBox(width: 6),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: _kAccentColor.withValues(alpha: 0.18),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: const Text(
                    'BETA',
                    style: TextStyle(
                      color: _kAccentColor,
                      fontSize: 9,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 0.6,
                    ),
                  ),
                ),
                const Spacer(),
                Text(
                  '${steps.length} ${steps.length == 1 ? 'Zug' : 'Züge'}',
                  style: TextStyle(color: _kTextColor.withValues(alpha: 0.5), fontSize: 12),
                ),
              ],
            ),
            const SizedBox(height: 8),
            ConstrainedBox(
              constraints: const BoxConstraints(maxHeight: 280),
              child: ListView.builder(
                shrinkWrap: true,
                itemCount: steps.length,
                itemBuilder: (context, index) => _buildBetaStepRow(index, steps[index], route),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBetaStepRow(int index, BetaStep step, List route) {
    final bool active = _selectedBetaStepIndex == index;
    final Map? grip =
        step.hold >= 0 && step.hold < route.length && route[step.hold] is Map
            ? route[step.hold] as Map
            : null;
    final Color holdColor =
        grip != null ? resolveDetectionColor(grip, _classColorMap) : _kAccentColor;
    final bool leftHand = step.hand == 'left' || step.hand == 'both';
    final bool rightHand = step.hand == 'right' || step.hand == 'both';

    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(8),
        onTap: () => _handleBetaStepTap(index, step),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          margin: const EdgeInsets.symmetric(vertical: 1),
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
          decoration: BoxDecoration(
            color: active ? _kAccentColor.withValues(alpha: 0.18) : Colors.transparent,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: active ? _kAccentColor.withValues(alpha: 0.55) : Colors.transparent,
            ),
          ),
          child: Row(
            children: [
              // Schritt-Chip im Stil der Overlay-Badges: heller Grund, dunkle
              // Schrift, Rand in der Grifffarbe.
              Container(
                constraints: const BoxConstraints(minWidth: 28),
                padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.92),
                  borderRadius: BorderRadius.circular(9),
                  border: Border.all(color: holdColor),
                ),
                child: Text(
                  step.action == 'top' ? 'TOP' : '${step.step}',
                  style: const TextStyle(
                    color: Color(0xFF1C1C1E),
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      _betaStepText(step),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        color: _kTextColor.withValues(alpha: active ? 1 : 0.8),
                        fontSize: 12.5,
                        fontWeight: active ? FontWeight.bold : FontWeight.normal,
                      ),
                    ),
                    if (step.hinweis != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 1),
                        child: Text(
                          step.hinweis!,
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            color: _kTextColor.withValues(alpha: 0.5),
                            fontSize: 10.5,
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              const SizedBox(width: 6),
              if (leftHand)
                Transform.flip(
                  flipX: true,
                  child: Icon(
                    Icons.back_hand_outlined,
                    size: 13,
                    color: _kTextColor.withValues(alpha: 0.6),
                  ),
                ),
              if (leftHand && rightHand) const SizedBox(width: 2),
              if (rightHand)
                Icon(Icons.back_hand_outlined, size: 13, color: _kTextColor.withValues(alpha: 0.6)),
            ],
          ),
        ),
      ),
    );
  }

  void _handleGripTap(Map det, int? routeIndex) {
    setState(() {
      _tappedGrip = det;
      _selectedBetaStepIndex = null;
      _betaStepGrip = null;
      if (routeIndex != null) {
        _selectedRouteIndex = routeIndex;
      }
    });
  }

  void _handleEmptyTap() {
    if (_selectedRouteIndex != null || _tappedGrip != null || _betaStepGrip != null) {
      setState(() {
        _selectedRouteIndex = null;
        _tappedGrip = null;
        _selectedBetaStepIndex = null;
        _betaStepGrip = null;
      });
    } else {
      _openFullscreen();
    }
  }

  void _handleBetaStepTap(int index, BetaStep step) {
    final int? sel = _selectedRouteIndex;
    if (sel == null || sel >= _routes.length || _routes[sel] is! List) {
      return;
    }
    final List route = _routes[sel] as List;
    final Map? grip =
        step.hold >= 0 && step.hold < route.length && route[step.hold] is Map
            ? route[step.hold] as Map
            : null;
    setState(() {
      if (_selectedBetaStepIndex == index) {
        _selectedBetaStepIndex = null;
        _betaStepGrip = null;
      } else {
        _selectedBetaStepIndex = index;
        _betaStepGrip = grip;
        _tappedGrip = null;
      }
    });
  }

  void _openFullscreen() {
    showDialog<void>(
      context: context,
      barrierColor: Colors.black.withValues(alpha: 0.9),
      builder: (_) => _DetectionFullscreen(
        imageBytes: _origBytes!,
        detections: _detections,
        routes: _routes,
        betas: _betas,
        originalWidth: _imgW,
        originalHeight: _imgH,
        classColorMap: _classColorMap,
        classNames: _classNames,
        detectionToRoute: _detectionToRoute,
        routeInfos: _routeInfos,
        hasSegmentation: _hasSegmentation,
        initialRouteIndex: _selectedRouteIndex,
        initialShowOverlay: _showOverlay,
        initialUseShapes: _useShapes,
        initialShowBeta: _showBeta,
        onSync: (routeIndex, showOverlay, useShapes, showBeta, grip) {
          if (!mounted) {
            return;
          }
          setState(() {
            _selectedRouteIndex = routeIndex;
            _showOverlay = showOverlay;
            _useShapes = useShapes;
            _showBeta = showBeta;
            _tappedGrip = grip;
            _selectedBetaStepIndex = null;
            _betaStepGrip = null;
          });
        },
      ),
    );
  }

  Widget _buildColorDistribution() {
    final int total = _detections.length;
    return Card(
      color: _kCardColor,
      elevation: 2,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.palette_outlined, color: _kAccentColor, size: 20),
                const SizedBox(width: 8),
                const Text(
                  'Erkannte Griffe',
                  style: TextStyle(color: _kTextColor, fontSize: 15, fontWeight: FontWeight.bold),
                ),
                const Spacer(),
                Text(
                  '$total gesamt',
                  style: TextStyle(color: _kTextColor.withValues(alpha: 0.5), fontSize: 12),
                ),
              ],
            ),
            const SizedBox(height: 12),
            if (_colorStats.isEmpty)
              Text('Keine Griffe erkannt.', style: TextStyle(color: _kTextColor.withValues(alpha: 0.7)))
            else
              ..._colorStats.map(
                (stat) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 5),
                  child: Row(
                    children: [
                      Container(
                        width: 14,
                        height: 14,
                        decoration: BoxDecoration(
                          color: stat.color,
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white.withValues(alpha: 0.25)),
                        ),
                      ),
                      const SizedBox(width: 10),
                      SizedBox(
                        width: 74,
                        child: Text(
                          stat.name,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(color: _kTextColor, fontSize: 13),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Container(
                          height: 6,
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.07),
                            borderRadius: BorderRadius.circular(3),
                          ),
                          alignment: Alignment.centerLeft,
                          child: FractionallySizedBox(
                            widthFactor: total == 0 ? 0 : (stat.count / total).clamp(0.02, 1).toDouble(),
                            heightFactor: 1,
                            child: DecoratedBox(
                              decoration: BoxDecoration(
                                color: stat.color,
                                borderRadius: BorderRadius.circular(3),
                              ),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        '${stat.count}× · Ø ${(stat.sumConf / stat.count * 100).toStringAsFixed(0)} %',
                        style: TextStyle(color: _kTextColor.withValues(alpha: 0.7), fontSize: 11),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

String _capitalize(String value) {
  return value.isEmpty ? value : value[0].toUpperCase() + value.substring(1);
}

String _handWord(String hand) {
  switch (hand) {
    case 'left':
      return 'linke Hand';
    case 'right':
      return 'rechte Hand';
    default:
      return 'beide Hände';
  }
}

/// Textzeile eines Zugfolge-Schritts, z. B. "1 · Start · linke Hand",
/// "2 · rechte Hand", "5 · Doppeln", "9 · Top".
String _betaStepText(BetaStep step) {
  switch (step.action) {
    case 'start':
      return '${step.step} · Start · ${_handWord(step.hand)}';
    case 'match':
      return '${step.step} · Doppeln';
    case 'top':
      return '${step.step} · Top';
    default:
      return '${step.step} · ${_handWord(step.hand)}';
  }
}

String _detectionColorName(Map det, Map<int, String> classNames) {
  final dynamic name = det["class_name"] ?? (det["color"] is Map ? det["color"]["name"] : null);
  if (name is String && name.isNotEmpty) {
    return _capitalize(name);
  }
  final int clsId = det["class"] is int ? det["class"] as int : 0;
  return _capitalize(classNames[clsId] ?? 'Unbekannt #$clsId');
}

class _RouteInfo {
  final int index;
  final Color color;
  final int gripCount;

  const _RouteInfo({required this.index, required this.color, required this.gripCount});
}

class _ColorStat {
  final String name;
  final Color color;
  int count = 0;
  double sumConf = 0;

  _ColorStat({required this.name, required this.color});
}

/// Routen-Auswahl: "Alle" plus ein Chip je Route mit Farbpunkt.
/// [wrap] = true bricht um (Seitenleiste), sonst horizontal scrollbar.
class _RouteChipBar extends StatelessWidget {
  final List<_RouteInfo> routes;
  final int? selectedIndex;
  final ValueChanged<int?> onSelected;
  final bool wrap;

  const _RouteChipBar({
    required this.routes,
    required this.selectedIndex,
    required this.onSelected,
    this.wrap = false,
  });

  @override
  Widget build(BuildContext context) {
    final List<Widget> chips = [
      _chip(label: 'Alle', active: selectedIndex == null, dotColor: null, onTap: () => onSelected(null)),
      for (final route in routes)
        _chip(
          label:
              'Route ${route.index + 1} · ${route.gripCount} ${route.gripCount == 1 ? 'Griff' : 'Griffe'}',
          active: selectedIndex == route.index,
          dotColor: route.color,
          onTap: () => onSelected(selectedIndex == route.index ? null : route.index),
        ),
    ];
    if (wrap) {
      return Wrap(spacing: 8, runSpacing: 8, children: chips);
    }
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: Row(
        children: [
          for (int i = 0; i < chips.length; i++) ...[
            if (i > 0) const SizedBox(width: 8),
            chips[i],
          ],
        ],
      ),
    );
  }

  Widget _chip({required String label, required bool active, Color? dotColor, required VoidCallback onTap}) {
    final Color borderColor = active ? (dotColor ?? _kAccentColor) : Colors.white.withValues(alpha: 0.10);
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
          decoration: BoxDecoration(
            color: active ? _kAccentColor.withValues(alpha: 0.22) : _kBackgroundColor,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: borderColor, width: active ? 1.4 : 1),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (dotColor != null) ...[
                Container(
                  width: 10,
                  height: 10,
                  decoration: BoxDecoration(
                    color: dotColor,
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white.withValues(alpha: 0.3)),
                  ),
                ),
                const SizedBox(width: 6),
              ] else ...[
                Icon(Icons.apps, size: 13, color: _kTextColor.withValues(alpha: active ? 0.9 : 0.6)),
                const SizedBox(width: 5),
              ],
              Text(
                label,
                style: TextStyle(
                  color: _kTextColor.withValues(alpha: active ? 1 : 0.75),
                  fontSize: 13,
                  fontWeight: active ? FontWeight.bold : FontWeight.normal,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Umschalter zwischen Segmentierungs-Formen und Bounding-Boxen.
class _ShapesToggle extends StatelessWidget {
  final bool useShapes;
  final ValueChanged<bool> onChanged;

  const _ShapesToggle({required this.useShapes, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(3),
      decoration: BoxDecoration(
        color: _kBackgroundColor,
        borderRadius: BorderRadius.circular(9),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _segment('Formen', useShapes, () => onChanged(true)),
          _segment('Boxen', !useShapes, () => onChanged(false)),
        ],
      ),
    );
  }

  Widget _segment(String label, bool active, VoidCallback onTap) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(7),
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
          decoration: BoxDecoration(
            color: active ? _kAccentColor.withValues(alpha: 0.30) : Colors.transparent,
            borderRadius: BorderRadius.circular(7),
          ),
          child: Text(
            label,
            style: TextStyle(
              color: active ? _kTextColor : _kTextColor.withValues(alpha: 0.45),
              fontSize: 12,
              fontWeight: active ? FontWeight.bold : FontWeight.normal,
            ),
          ),
        ),
      ),
    );
  }
}

/// Umschalter für die Zugvorschläge (Beta): Zugfolge statt Routenlinie.
/// Nur sichtbar, wenn eine Route ausgewählt ist und Beta-Daten existieren.
class _BetaToggle extends StatelessWidget {
  final bool active;
  final ValueChanged<bool> onChanged;

  const _BetaToggle({required this.active, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: active ? 'Zugfolge ausblenden' : 'Zugfolge (Beta) anzeigen',
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(9),
          onTap: () => onChanged(!active),
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 150),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              color: active ? _kAccentColor.withValues(alpha: 0.30) : _kBackgroundColor,
              borderRadius: BorderRadius.circular(9),
              border: Border.all(
                color: active
                    ? _kAccentColor.withValues(alpha: 0.8)
                    : Colors.white.withValues(alpha: 0.08),
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  Icons.moving,
                  size: 13,
                  color: active ? _kTextColor : _kTextColor.withValues(alpha: 0.45),
                ),
                const SizedBox(width: 5),
                Text(
                  'Beta',
                  style: TextStyle(
                    color: active ? _kTextColor : _kTextColor.withValues(alpha: 0.45),
                    fontSize: 12,
                    fontWeight: active ? FontWeight.bold : FontWeight.normal,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// Kleines Info-Panel zum angetippten Griff (Farbname, Typ, Confidence, Route).
class _GripInfoCard extends StatelessWidget {
  final Map det;
  final Color color;
  final String colorName;
  final int? routeIndex;
  final VoidCallback onClose;

  const _GripInfoCard({
    required this.det,
    required this.color,
    required this.colorName,
    required this.routeIndex,
    required this.onClose,
  });

  @override
  Widget build(BuildContext context) {
    final bool isVolume = det["det_class"] == "volume";
    final double conf = ((det["confidence"] as num?) ?? 0).toDouble();
    final String details =
        'Confidence ${(conf * 100).toStringAsFixed(0)} %${routeIndex != null ? ' · Route ${routeIndex! + 1}' : ''}';

    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.92, end: 1),
      duration: const Duration(milliseconds: 160),
      curve: Curves.easeOut,
      builder: (context, value, child) => Opacity(
        opacity: ((value - 0.92) / 0.08).clamp(0.0, 1.0),
        child: Transform.scale(scale: value, child: child),
      ),
      child: Material(
        color: Colors.transparent,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          decoration: BoxDecoration(
            color: _kCardColor.withValues(alpha: 0.96),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: _kAccentColor.withValues(alpha: 0.45)),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.45),
                blurRadius: 14,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 16,
                height: 16,
                decoration: BoxDecoration(
                  color: color,
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white.withValues(alpha: 0.35)),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '$colorName · ${isVolume ? 'Volume' : 'Griff'}',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(color: _kTextColor, fontSize: 13, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      details,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(color: _kTextColor.withValues(alpha: 0.6), fontSize: 11),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 6),
              InkWell(
                borderRadius: BorderRadius.circular(10),
                onTap: onClose,
                child: Padding(
                  padding: const EdgeInsets.all(4),
                  child: Icon(Icons.close, size: 15, color: _kTextColor.withValues(alpha: 0.55)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Vollbild-Ansicht der KI-Detektion: Overlay zoomt mit, Routen-Chips und
/// Griff-Info liegen als nicht zoomende Leiste darüber.
class _DetectionFullscreen extends StatefulWidget {
  final Uint8List imageBytes;
  final List<dynamic> detections;
  final List<dynamic> routes;
  final List<List<BetaStep>?> betas;
  final double originalWidth;
  final double originalHeight;
  final Map<int, Color> classColorMap;
  final Map<int, String> classNames;
  final Map<String, int> detectionToRoute;
  final List<_RouteInfo> routeInfos;
  final bool hasSegmentation;
  final int? initialRouteIndex;
  final bool initialShowOverlay;
  final bool initialUseShapes;
  final bool initialShowBeta;
  final void Function(
    int? routeIndex,
    bool showOverlay,
    bool useShapes,
    bool showBeta,
    Map? grip,
  ) onSync;

  const _DetectionFullscreen({
    required this.imageBytes,
    required this.detections,
    required this.routes,
    required this.betas,
    required this.originalWidth,
    required this.originalHeight,
    required this.classColorMap,
    required this.classNames,
    required this.detectionToRoute,
    required this.routeInfos,
    required this.hasSegmentation,
    required this.initialRouteIndex,
    required this.initialShowOverlay,
    required this.initialUseShapes,
    required this.initialShowBeta,
    required this.onSync,
  });

  @override
  State<_DetectionFullscreen> createState() => _DetectionFullscreenState();
}

class _DetectionFullscreenState extends State<_DetectionFullscreen> {
  late int? _routeIndex = widget.initialRouteIndex;
  late bool _showOverlay = widget.initialShowOverlay;
  late bool _useShapes = widget.initialUseShapes;
  late bool _showBeta = widget.initialShowBeta;
  Map? _grip;

  /// Zugfolge der im Vollbild ausgewählten Route, sonst null.
  List<BetaStep>? get _selectedBeta {
    final int? sel = _routeIndex;
    if (sel == null || sel < 0 || sel >= widget.betas.length) {
      return null;
    }
    return widget.betas[sel];
  }

  void _update(VoidCallback fn) {
    setState(fn);
    widget.onSync(_routeIndex, _showOverlay, _useShapes, _showBeta, _grip);
  }

  @override
  Widget build(BuildContext context) {
    return FullscreenViewer(
      title: 'Boulderwand',
      actions: [
        if (_selectedBeta != null)
          Padding(
            padding: const EdgeInsets.only(right: 6),
            child: _BetaToggle(
              active: _showBeta,
              onChanged: (value) => _update(() => _showBeta = value),
            ),
          ),
        if (widget.hasSegmentation)
          Padding(
            padding: const EdgeInsets.only(right: 6),
            child: _ShapesToggle(
              useShapes: _useShapes,
              onChanged: (value) => _update(() => _useShapes = value),
            ),
          ),
        IconButton(
          tooltip: _showOverlay ? 'Overlay ausblenden' : 'Overlay einblenden',
          onPressed: () => _update(() => _showOverlay = !_showOverlay),
          icon: Icon(
            _showOverlay ? Icons.visibility : Icons.visibility_off,
            color: _showOverlay ? _kAccentColor : Colors.white38,
            size: 20,
          ),
        ),
      ],
      footer: (widget.routeInfos.isEmpty && _grip == null)
          ? null
          : Container(
              padding: const EdgeInsets.fromLTRB(16, 28, 16, 14),
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [Color(0x00000000), Color(0xB3000000)],
                ),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (_grip != null)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 10),
                      child: ConstrainedBox(
                        constraints: const BoxConstraints(maxWidth: 380),
                        child: _GripInfoCard(
                          det: _grip!,
                          color: resolveDetectionColor(_grip!, widget.classColorMap),
                          colorName: _detectionColorName(_grip!, widget.classNames),
                          routeIndex: widget.detectionToRoute[_grip!["bbox"].toString()],
                          onClose: () => _update(() => _grip = null),
                        ),
                      ),
                    ),
                  if (widget.routeInfos.isNotEmpty)
                    _RouteChipBar(
                      routes: widget.routeInfos,
                      selectedIndex: _routeIndex,
                      onSelected: (index) => _update(() {
                        _routeIndex = index;
                        _grip = null;
                      }),
                    ),
                ],
              ),
            ),
      child: DetectionOverlayInteractive(
        imageBytes: widget.imageBytes,
        detections: widget.detections,
        routes: widget.routes,
        betas: widget.betas,
        showBeta: _showBeta,
        originalWidth: widget.originalWidth,
        originalHeight: widget.originalHeight,
        classColorMap: widget.classColorMap,
        detectionToRoute: widget.detectionToRoute,
        selectedRouteIndex: _routeIndex,
        showOverlay: _showOverlay,
        useShapes: _useShapes,
        highlightedDetection: _grip,
        onGripTapped: (det, routeIndex) => _update(() {
          _grip = det;
          if (routeIndex != null) {
            _routeIndex = routeIndex;
          }
        }),
        onEmptyTapped: () => _update(() {
          _routeIndex = null;
          _grip = null;
        }),
      ),
    );
  }
}
