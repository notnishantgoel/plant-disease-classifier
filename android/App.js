/**
 * PlantGuard AI — React Native (Expo) App
 *
 * Setup:
 *   cd android && npm install && npx expo start --android
 *
 * Update API_BASE_URL in src/constants/models.js to your machine's local IP
 * when testing on a physical device (e.g. 'http://192.168.x.x:8000').
 */

import React, { useState } from 'react'
import {
  View, Text, Image, TouchableOpacity, ScrollView,
  ActivityIndicator, StyleSheet, Alert, Platform,
  SafeAreaView, StatusBar,
} from 'react-native'
import * as ImagePicker from 'expo-image-picker'

import ModelSelector from './src/components/ModelSelector'
import ResultCard from './src/components/ResultCard'
import { MODELS, API_BASE_URL } from './src/constants/models'

export default function App() {
  const [selected, setSelected] = useState(MODELS[0])
  const [imageUri, setImageUri] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  // ── Image picker helpers ──────────────────────────────────────
  const pickFromGallery = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync()
    if (status !== 'granted') {
      Alert.alert('Permission required', 'Photo library access is needed to select images.')
      return
    }
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 0.9,
    })
    if (!res.canceled) {
      setImageUri(res.assets[0].uri)
      setResult(null)
    }
  }

  const pickFromCamera = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync()
    if (status !== 'granted') {
      Alert.alert('Permission required', 'Camera access is needed to take photos.')
      return
    }
    const res = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      quality: 0.9,
    })
    if (!res.canceled) {
      setImageUri(res.assets[0].uri)
      setResult(null)
    }
  }

  // ── Inference ─────────────────────────────────────────────────
  const handlePredict = async () => {
    if (!imageUri) return
    setLoading(true)
    setResult(null)

    try {
      const form = new FormData()
      const filename = imageUri.split('/').pop()
      const ext = filename.split('.').pop().toLowerCase()
      const mimeType = ext === 'png' ? 'image/png' : 'image/jpeg'

      form.append('file', { uri: imageUri, name: filename, type: mimeType })
      form.append('model_id', selected.id)

      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: form,
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail ?? `Server error ${res.status}`)
      }

      setResult(await res.json())
    } catch (err) {
      Alert.alert('Error', err.message)
    } finally {
      setLoading(false)
    }
  }

  // ── UI ────────────────────────────────────────────────────────
  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="light-content" backgroundColor="#1b4332" />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerEmoji}>🌿</Text>
        <View>
          <Text style={styles.headerTitle}>PlantGuard AI</Text>
          <Text style={styles.headerSub}>Plant Disease Classifier</Text>
        </View>
      </View>

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled"
      >
        {/* Model Selector */}
        <ModelSelector models={MODELS} selected={selected} onSelect={(m) => { setSelected(m); setResult(null) }} />

        {/* Image area */}
        <View style={styles.imageBox}>
          {imageUri ? (
            <Image source={{ uri: imageUri }} style={styles.image} resizeMode="contain" />
          ) : (
            <View style={styles.imagePlaceholder}>
              <Text style={styles.placeholderIcon}>🌿</Text>
              <Text style={styles.placeholderText}>No image selected</Text>
            </View>
          )}
        </View>

        {/* Pick buttons */}
        <View style={styles.pickRow}>
          <TouchableOpacity style={styles.pickBtn} onPress={pickFromGallery} activeOpacity={0.75}>
            <Text style={styles.pickBtnText}>📁  Gallery</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.pickBtn} onPress={pickFromCamera} activeOpacity={0.75}>
            <Text style={styles.pickBtnText}>📷  Camera</Text>
          </TouchableOpacity>
        </View>

        {/* Analyze button */}
        <TouchableOpacity
          style={[styles.analyzeBtn, (!imageUri || loading) && styles.analyzeBtnDisabled]}
          onPress={handlePredict}
          disabled={!imageUri || loading}
          activeOpacity={0.8}
        >
          {loading
            ? <ActivityIndicator color="#fff" />
            : <Text style={styles.analyzeBtnText}>Analyze Plant  →</Text>
          }
        </TouchableOpacity>

        {/* Result */}
        {result && <ResultCard result={result} />}

        {/* Footer */}
        <Text style={styles.footer}>
          4 classes · 4 models · trained from scratch
        </Text>
      </ScrollView>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#1b4332' },
  scroll: { flex: 1, backgroundColor: '#f0fdf4' },
  content: { padding: 18, gap: 16, paddingBottom: 40 },

  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    padding: 16,
    backgroundColor: '#1b4332',
  },
  headerEmoji: { fontSize: 32 },
  headerTitle: { fontSize: 20, fontWeight: '700', color: '#fff' },
  headerSub:   { fontSize: 12, color: 'rgba(255,255,255,.65)', marginTop: 2 },

  imageBox: {
    height: 240,
    borderRadius: 14,
    borderWidth: 2,
    borderColor: '#b7e4c7',
    borderStyle: 'dashed',
    overflow: 'hidden',
    backgroundColor: '#fff',
  },
  image: { width: '100%', height: '100%' },
  imagePlaceholder: {
    flex: 1, alignItems: 'center', justifyContent: 'center', gap: 10,
  },
  placeholderIcon: { fontSize: 48 },
  placeholderText: { fontSize: 14, color: '#94a3b8' },

  pickRow: { flexDirection: 'row', gap: 12 },
  pickBtn: {
    flex: 1,
    padding: 13,
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: '#b7e4c7',
    backgroundColor: '#fff',
    alignItems: 'center',
  },
  pickBtnText: { fontSize: 14, fontWeight: '600', color: '#2d6a4f' },

  analyzeBtn: {
    padding: 15,
    borderRadius: 12,
    backgroundColor: '#2d6a4f',
    alignItems: 'center',
  },
  analyzeBtnDisabled: { opacity: 0.45 },
  analyzeBtnText: { fontSize: 16, fontWeight: '700', color: '#fff' },

  footer: {
    textAlign: 'center', fontSize: 11.5, color: '#94a3b8', marginTop: 8,
  },
})
