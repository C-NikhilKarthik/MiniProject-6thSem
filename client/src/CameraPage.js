import React, { useState, useEffect } from 'react';
import { View, TouchableOpacity, Image, StyleSheet } from 'react-native';
import { Entypo } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';

function CameraPage() {
    const [selectedImage, setSelectedImage] = useState(null);

    useEffect(() => {
        (async () => {
            const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
            if (status !== 'granted') {
                alert('Sorry, we need camera roll permissions to make this work!');
            }
        })();
    }, []);

    const selectImage = async () => {
        try {
            const result = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ImagePicker.MediaTypeOptions.Images,
                allowsEditing: true,
                aspect: [4, 3],
                quality: 1,
            });

            if (!result.cancelled) {
                setSelectedImage(result.uri);
            }
        } catch (error) {
            console.error('Error selecting image:', error);
        }
    };

    const uploadImage = async () => {
        try {
            const formData = new FormData();
            formData.append('image', {
                uri: selectedImage,
                type: 'image/jpeg',
                name: 'photo.jpg'
            });

            const response = await axios.post('http://10.0.3.61:5000/predict', formData);
            const responseData = response.data;

            // Handle the response as needed
        } catch (error) {
            console.error('Error uploading image:', error);
        }
    };

    return (
        <View style={styles.container}>
            {selectedImage && <Image source={{ uri: selectedImage }} style={styles.image} />}
            <TouchableOpacity style={styles.button} onPress={selectImage}>
                <Entypo name='folder-images' size={24} color='white' />
            </TouchableOpacity>
            {selectedImage && (
                <TouchableOpacity style={styles.button} onPress={uploadImage}>
                    <Entypo name='upload' size={24} color='white' />
                </TouchableOpacity>
            )}
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    image: {
        width: 300,
        height: 300,
        resizeMode: 'cover',
        marginBottom: 20,
    },
    button: {
        backgroundColor: 'rgba(0,0,0,0.5)',
        padding: 10,
        borderRadius: 10,
        marginVertical: 10,
    },
});

export default CameraPage;
