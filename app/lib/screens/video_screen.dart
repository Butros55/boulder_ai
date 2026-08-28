import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:video_player/video_player.dart';

const Color _kBackgroundColor = Color(0xFF1C1C1E);
const Color _kCardColor = Color(0xFF2C2C2E);
const Color _kAccentColor = Color(0xFF7F5AF0);
const Color _kTextColor = Colors.white;

/// Lebenszyklus eines Video-Jobs aus Sicht des Screens.
enum _VideoPhase { uploading, queued, running, downloading, done, error }

/// Lädt ein Video zur Analyse hoch, verfolgt den Job-Fortschritt per Polling
/// und zeigt das annotierte Ergebnis an.
///
/// Auf Windows-Desktop wird das fertige MP4 heruntergeladen und im
/// Standardplayer geöffnet, weil video_player dort nicht unterstützt wird.
/// Auf Mobile/Web spielt ein Inline-Player das Ergebnis direkt ab.
class VideoScreen extends StatefulWidget {
  /// Pfad zur Videodatei (Mobile/Desktop).
  final String? videoPath;

  /// Videodaten im Speicher (Web).
  final Uint8List? videoBytes;

  final String filename;

  const VideoScreen({
    super.key,
    this.videoPath,
    this.videoBytes,
    this.filename = 'video.mp4',
  });

  @override
  State<VideoScreen> createState() => _VideoScreenState();
}

class _VideoScreenState extends State<VideoScreen> {
  static const String _baseUrl = 'http://127.0.0.1:5000';
  final FlutterSecureStorage _storage = const FlutterSecureStorage();

  _VideoPhase _phase = _VideoPhase.uploading;
  double _progress = 0;
  double? _downloadProgress;
  String? _errorMessage;
  String? _jobId;
  String? _token;
  int? _frames;
  num? _dauer;

  Timer? _pollTimer;
  bool _pollInFlight = false;

  String? _localFilePath;
  VideoPlayerController? _playerController;

  bool get _isWindowsDesktop => !kIsWeb && Platform.isWindows;

  @override
  void initState() {
    super.initState();
    _startUpload();
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _playerController?.dispose();
    super.dispose();
  }

  void _stopPolling() {
    _pollTimer?.cancel();
    _pollTimer = null;
  }

  void _fail(String message) {
    _stopPolling();
    if (!mounted) return;
    setState(() {
      _phase = _VideoPhase.error;
      _errorMessage = message;
    });
  }

  /// Versucht, aus einem JSON-Fehlerkörper die Meldung unter "fehler" zu lesen.
  String? _serverError(String body) {
    try {
      final dynamic data = jsonDecode(body);
      if (data is Map && data['fehler'] is String) {
        return data['fehler'] as String;
      }
    } catch (_) {
      // Kein JSON, dann gibt es keine Detailmeldung.
    }
    return null;
  }

  Future<void> _startUpload() async {
    try {
      final String? token = await _storage.read(key: 'jwt_token');
      if (token == null) {
        _fail('Deine Sitzung ist abgelaufen. Bitte logge dich erneut ein.');
        return;
      }
      _token = token;

      final http.MultipartRequest request = http.MultipartRequest(
        'POST',
        Uri.parse('$_baseUrl/process_video'),
      );
      request.headers['Authorization'] = 'Bearer $token';

      if (kIsWeb) {
        final Uint8List? bytes = widget.videoBytes;
        if (bytes == null) {
          _fail('Es wurden keine Videodaten übergeben.');
          return;
        }
        request.files.add(
          http.MultipartFile.fromBytes('video', bytes, filename: widget.filename),
        );
      } else {
        final String? path = widget.videoPath;
        if (path == null) {
          _fail('Es wurde keine Videodatei übergeben.');
          return;
        }
        request.files.add(await http.MultipartFile.fromPath('video', path));
      }

      final http.StreamedResponse response = await request.send();
      final String body = await response.stream.bytesToString();
      if (!mounted) return;

      if (response.statusCode == 201) {
        final dynamic data = jsonDecode(body);
        final String? jobId = data is Map ? data['job_id'] as String? : null;
        if (jobId == null) {
          _fail('Die Antwort des Servers enthielt keine Job-ID.');
          return;
        }
        setState(() {
          _jobId = jobId;
          _phase = _VideoPhase.queued;
          _progress = 0;
        });
        _pollTimer = Timer.periodic(
          const Duration(seconds: 2),
          (_) => _pollStatus(),
        );
      } else if (response.statusCode == 401) {
        _fail('Deine Sitzung ist abgelaufen. Bitte logge dich erneut ein.');
      } else {
        _fail(
          _serverError(body) ??
              'Upload fehlgeschlagen (Status ${response.statusCode}).',
        );
      }
    } catch (e) {
      debugPrint('Fehler beim Video-Upload: $e');
      _fail('Das Video konnte nicht hochgeladen werden. Läuft der Server?');
    }
  }

  Future<void> _pollStatus() async {
    if (_pollInFlight || _jobId == null || _token == null) return;
    _pollInFlight = true;
    try {
      final http.Response response = await http.get(
        Uri.parse('$_baseUrl/process_video/$_jobId'),
        headers: {'Authorization': 'Bearer $_token'},
      );
      if (!mounted) return;

      if (response.statusCode == 401) {
        _fail('Deine Sitzung ist abgelaufen. Bitte logge dich erneut ein.');
        return;
      }
      if (response.statusCode != 200) {
        // Einzelne Fehlversuche überbrücken, der nächste Poll kommt in 2 s.
        debugPrint('Statusabruf fehlgeschlagen: ${response.statusCode}');
        return;
      }

      final dynamic data = jsonDecode(response.body);
      if (data is! Map) return;
      final String status = data['status'] as String? ?? 'error';
      final double progress = ((data['progress'] as num?) ?? 0).toDouble();

      if (status == 'queued') {
        setState(() {
          _phase = _VideoPhase.queued;
          _progress = progress;
        });
      } else if (status == 'running') {
        setState(() {
          _phase = _VideoPhase.running;
          _progress = progress;
        });
      } else if (status == 'done') {
        _stopPolling();
        _frames = (data['frames'] as num?)?.toInt();
        _dauer = data['dauer'] as num?;
        await _prepareResult();
      } else {
        _fail((data['fehler'] as String?) ?? 'Die Verarbeitung ist fehlgeschlagen.');
      }
    } catch (e) {
      debugPrint('Fehler beim Statusabruf: $e');
    } finally {
      _pollInFlight = false;
    }
  }

  Future<void> _prepareResult() async {
    final Uri resultUri = Uri.parse('$_baseUrl/process_video/$_jobId/result');
    if (_isWindowsDesktop) {
      if (!mounted) return;
      setState(() {
        _phase = _VideoPhase.downloading;
        _downloadProgress = null;
      });
      await _downloadResult(resultUri);
      return;
    }

    try {
      final VideoPlayerController controller = VideoPlayerController.networkUrl(
        resultUri,
        httpHeaders: {'Authorization': 'Bearer $_token'},
      );
      _playerController = controller;
      await controller.initialize();
      if (!mounted) return;
      setState(() => _phase = _VideoPhase.done);
    } catch (e) {
      debugPrint('Fehler beim Initialisieren des Players: $e');
      _fail('Das Ergebnisvideo konnte nicht geladen werden.');
    }
  }

  /// Lädt das fertige MP4 mit Bearer-Header in eine lokale Datei
  /// (Windows-Weg, weil video_player dort nicht unterstützt wird).
  Future<void> _downloadResult(Uri uri) async {
    final http.Client client = http.Client();
    try {
      final http.Request request = http.Request('GET', uri);
      request.headers['Authorization'] = 'Bearer $_token';
      final http.StreamedResponse response = await client.send(request);
      if (response.statusCode != 200) {
        _fail(
          'Das Ergebnis konnte nicht geladen werden (Status ${response.statusCode}).',
        );
        return;
      }

      Directory dir;
      try {
        dir = await getTemporaryDirectory();
      } catch (_) {
        dir = Directory.systemTemp;
      }
      final File file = File(
        '${dir.path}${Platform.pathSeparator}boulder_ai_$_jobId.mp4',
      );
      final IOSink sink = file.openWrite();
      final int? total = response.contentLength;
      int received = 0;
      try {
        await for (final List<int> chunk in response.stream) {
          sink.add(chunk);
          received += chunk.length;
          if (total != null && total > 0 && mounted) {
            final double p = received / total;
            if (p - (_downloadProgress ?? 0) >= 0.01 || p >= 1) {
              setState(() => _downloadProgress = p);
            }
          }
        }
        await sink.flush();
      } finally {
        await sink.close();
      }

      if (!mounted) return;
      setState(() {
        _localFilePath = file.path;
        _phase = _VideoPhase.done;
      });
    } catch (e) {
      debugPrint('Fehler beim Herunterladen des Ergebnisvideos: $e');
      _fail('Das Ergebnisvideo konnte nicht heruntergeladen werden.');
    } finally {
      client.close();
    }
  }

  /// Öffnet die heruntergeladene Datei im Standardplayer (nur Windows).
  Future<void> _openInPlayer() async {
    final String? path = _localFilePath;
    if (path == null) return;
    try {
      await Process.start('explorer', [path]);
    } catch (e) {
      debugPrint('Fehler beim Öffnen des Players: $e');
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'Das Video konnte nicht im Player geöffnet werden.',
            style: TextStyle(color: Colors.white),
          ),
          backgroundColor: Colors.red,
          duration: Duration(seconds: 3),
        ),
      );
    }
  }

  String _formatDuration(Duration d) {
    final String minutes = d.inMinutes.toString();
    final String seconds = (d.inSeconds % 60).toString().padLeft(2, '0');
    return '$minutes:$seconds';
  }

  String? get _statsLine {
    final List<String> parts = [];
    if (_frames != null) {
      parts.add('Frames: $_frames');
    }
    final num? dauer = _dauer;
    if (dauer != null) {
      parts.add('Dauer: ${dauer.toStringAsFixed(1)} s');
    }
    return parts.isEmpty ? null : parts.join('  ·  ');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _kBackgroundColor,
      appBar: AppBar(
        backgroundColor: _kBackgroundColor,
        title: const Text('Video-Analyse', style: TextStyle(color: _kTextColor)),
        iconTheme: const IconThemeData(color: _kTextColor),
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 560),
            child: Card(
              color: _kCardColor,
              elevation: 2,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: _buildContent(),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildContent() {
    switch (_phase) {
      case _VideoPhase.uploading:
        return _buildUploading();
      case _VideoPhase.queued:
      case _VideoPhase.running:
        return _buildProcessing();
      case _VideoPhase.downloading:
        return _buildDownloading();
      case _VideoPhase.done:
        return _isWindowsDesktop ? _buildDoneWindows() : _buildDonePlayer();
      case _VideoPhase.error:
        return _buildError();
    }
  }

  Widget _buildHint() {
    return Text(
      'Hinweis: Es werden maximal die ersten 60 Sekunden des Videos verarbeitet.',
      textAlign: TextAlign.center,
      style: TextStyle(color: _kTextColor.withValues(alpha: 0.6), fontSize: 12),
    );
  }

  Widget _buildCancelButton({bool jobRunning = false}) {
    return Column(
      children: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: Text(
            'Abbrechen',
            style: TextStyle(color: _kTextColor.withValues(alpha: 0.7)),
          ),
        ),
        if (jobRunning)
          Text(
            'Die Verarbeitung läuft auf dem Server weiter.',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: _kTextColor.withValues(alpha: 0.45),
              fontSize: 11,
            ),
          ),
      ],
    );
  }

  Widget _buildAccentButton({
    required String label,
    required IconData icon,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 20),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [_kAccentColor.withValues(alpha: 0.8), _kAccentColor],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: _kTextColor, size: 20),
            const SizedBox(width: 8),
            Text(
              label,
              style: const TextStyle(
                color: _kTextColor,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildUploading() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        const CircularProgressIndicator(
          valueColor: AlwaysStoppedAnimation<Color>(_kAccentColor),
        ),
        const SizedBox(height: 20),
        const Text(
          'Video wird hochgeladen ...',
          style: TextStyle(
            color: _kTextColor,
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          widget.filename,
          textAlign: TextAlign.center,
          overflow: TextOverflow.ellipsis,
          style: TextStyle(
            color: _kTextColor.withValues(alpha: 0.7),
            fontSize: 13,
          ),
        ),
        const SizedBox(height: 16),
        _buildHint(),
        const SizedBox(height: 16),
        _buildCancelButton(),
      ],
    );
  }

  Widget _buildProcessing() {
    final bool queued = _phase == _VideoPhase.queued;
    final int percent = (_progress.clamp(0.0, 1.0) * 100).round();
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(
          queued ? Icons.hourglass_top_rounded : Icons.movie_filter_rounded,
          color: _kAccentColor,
          size: 40,
        ),
        const SizedBox(height: 16),
        Text(
          queued ? 'In der Warteschlange' : 'Verarbeitung läuft',
          style: const TextStyle(
            color: _kTextColor,
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 20),
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: LinearProgressIndicator(
            value: _progress.clamp(0.0, 1.0),
            minHeight: 10,
            backgroundColor: _kBackgroundColor,
            valueColor: const AlwaysStoppedAnimation<Color>(_kAccentColor),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          '$percent %',
          style: TextStyle(
            color: _kTextColor.withValues(alpha: 0.8),
            fontSize: 13,
          ),
        ),
        const SizedBox(height: 16),
        _buildHint(),
        const SizedBox(height: 16),
        _buildCancelButton(jobRunning: true),
      ],
    );
  }

  Widget _buildDownloading() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        const Icon(Icons.download_rounded, color: _kAccentColor, size: 40),
        const SizedBox(height: 16),
        const Text(
          'Ergebnis wird geladen ...',
          style: TextStyle(
            color: _kTextColor,
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 20),
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: LinearProgressIndicator(
            value: _downloadProgress,
            minHeight: 10,
            backgroundColor: _kBackgroundColor,
            valueColor: const AlwaysStoppedAnimation<Color>(_kAccentColor),
          ),
        ),
        const SizedBox(height: 16),
        _buildCancelButton(),
      ],
    );
  }

  Widget _buildError() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        const Icon(Icons.error_outline, color: Colors.redAccent, size: 48),
        const SizedBox(height: 16),
        const Text(
          'Fehler bei der Video-Analyse',
          style: TextStyle(
            color: _kTextColor,
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          _errorMessage ?? 'Unbekannter Fehler.',
          textAlign: TextAlign.center,
          style: TextStyle(
            color: _kTextColor.withValues(alpha: 0.7),
            fontSize: 13,
          ),
        ),
        const SizedBox(height: 24),
        _buildAccentButton(
          label: 'Zurück',
          icon: Icons.arrow_back_rounded,
          onTap: () => Navigator.of(context).pop(),
        ),
      ],
    );
  }

  Widget _buildDoneWindows() {
    final String? path = _localFilePath;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        const Icon(
          Icons.check_circle_outline,
          color: Colors.greenAccent,
          size: 48,
        ),
        const SizedBox(height: 16),
        const Text(
          'Analyse abgeschlossen',
          style: TextStyle(
            color: _kTextColor,
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        if (_statsLine != null) ...[
          const SizedBox(height: 8),
          Text(
            _statsLine!,
            style: TextStyle(
              color: _kTextColor.withValues(alpha: 0.7),
              fontSize: 13,
            ),
          ),
        ],
        if (path != null) ...[
          const SizedBox(height: 20),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: _kBackgroundColor,
              borderRadius: BorderRadius.circular(10),
            ),
            child: SelectableText(
              path,
              style: TextStyle(
                color: _kTextColor.withValues(alpha: 0.8),
                fontSize: 12,
              ),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Speichern unter: Die Datei liegt unter dem angezeigten Pfad '
            'und kann von dort kopiert werden.',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: _kTextColor.withValues(alpha: 0.55),
              fontSize: 12,
            ),
          ),
          const SizedBox(height: 20),
          _buildAccentButton(
            label: 'Video im Player öffnen',
            icon: Icons.play_circle_outline_rounded,
            onTap: _openInPlayer,
          ),
        ],
        const SizedBox(height: 12),
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: Text(
            'Fertig',
            style: TextStyle(color: _kTextColor.withValues(alpha: 0.7)),
          ),
        ),
      ],
    );
  }

  Widget _buildDonePlayer() {
    final VideoPlayerController? controller = _playerController;
    if (controller == null || !controller.value.isInitialized) {
      return _buildError();
    }
    final double aspectRatio =
        controller.value.aspectRatio > 0 ? controller.value.aspectRatio : 16 / 9;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        const Text(
          'Analyse abgeschlossen',
          style: TextStyle(
            color: _kTextColor,
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        if (_statsLine != null) ...[
          const SizedBox(height: 8),
          Text(
            _statsLine!,
            style: TextStyle(
              color: _kTextColor.withValues(alpha: 0.7),
              fontSize: 13,
            ),
          ),
        ],
        const SizedBox(height: 16),
        ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: AspectRatio(
            aspectRatio: aspectRatio,
            child: VideoPlayer(controller),
          ),
        ),
        const SizedBox(height: 12),
        VideoProgressIndicator(
          controller,
          allowScrubbing: true,
          colors: const VideoProgressColors(
            playedColor: _kAccentColor,
            bufferedColor: Colors.white24,
            backgroundColor: Colors.white10,
          ),
        ),
        const SizedBox(height: 8),
        ValueListenableBuilder<VideoPlayerValue>(
          valueListenable: controller,
          builder: (context, value, _) {
            return Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                IconButton(
                  onPressed: () {
                    value.isPlaying ? controller.pause() : controller.play();
                  },
                  icon: Icon(
                    value.isPlaying
                        ? Icons.pause_rounded
                        : Icons.play_arrow_rounded,
                    color: _kTextColor,
                    size: 32,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  '${_formatDuration(value.position)} / '
                  '${_formatDuration(value.duration)}',
                  style: TextStyle(
                    color: _kTextColor.withValues(alpha: 0.7),
                    fontSize: 13,
                  ),
                ),
              ],
            );
          },
        ),
        const SizedBox(height: 8),
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: Text(
            'Fertig',
            style: TextStyle(color: _kTextColor.withValues(alpha: 0.7)),
          ),
        ),
      ],
    );
  }
}
